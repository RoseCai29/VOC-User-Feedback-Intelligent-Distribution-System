"""
分类评估 —— 准确率、宏观 F1、混淆矩阵，以及级联专属指标。

级联指标:
  coverage              小模型自动处理占比
  escalation_rate       升级给大模型/规则的比例
  precision_at_coverage 小模型直出部分的准确率(必须单独看，防止被 LLM 兜底掩盖)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .tools import classify, classify_semantic, classify_auto

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"


def load_eval_set(path: Path | str = EVAL_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["examples"]


def load_dataset(
    path: Path | str,
    text_field: str | None = None,
    category_field: str | None = None,
    sub_category_field: str | None = None,
    source_field: str | None = None,
) -> list[dict]:
    """加载标注数据集：JSONL / JSON / CSV / TSV，字段名自动识别或显式指定。"""
    from .classifier.dataset import load_examples
    return load_examples(
        path, text_field, category_field, sub_category_field, source_field,
    )


def label_rules(text: str, source: str = "社区") -> dict:
    return json.loads(classify(text, source))


def label_semantic(text: str, source: str = "社区") -> dict:
    return json.loads(classify_semantic(text, source))


def label_cascade(text: str, source: str = "社区") -> dict:
    return json.loads(classify_auto(text, source))


def _normalize(label_fn, text: str, source: str) -> tuple[str, str, str, bool]:
    """兼容两种标注函数：返回 dict(含 engine) 或 (category, sub_category) 元组。"""
    out = label_fn(text, source)
    if isinstance(out, dict):
        return (
            out.get("category", ""),
            out.get("sub_category", ""),
            out.get("engine", "unknown"),
            bool(out.get("escalated", False)),
        )
    pred_c, pred_s = out
    return pred_c, pred_s, "unknown", False


def _bootstrap_ci(flags: list[int], n_boot: int = 1000, seed: int = 42) -> list[float]:
    """对准确率做 bootstrap 置信区间，避免小样本单点数字被过度解读。"""
    n = len(flags)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(
        sum(flags[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4)]


def evaluate(label_fn, examples: list[dict]) -> dict:
    """对标注集计算 level-1/level-2 准确率、宏观 F1、混淆矩阵与级联指标。"""
    cats = sorted({e["category"] for e in examples})
    conf: dict[str, defaultdict] = {c: defaultdict(int) for c in cats}
    l1 = l2 = 0
    l1_flags: list[int] = []
    l2_flags: list[int] = []
    mis: list[dict] = []
    engines: dict[str, int] = defaultdict(int)
    covered = covered_l1 = covered_l2 = escalated = 0

    for e in examples:
        gold_c, gold_s = e["category"], e["sub_category"]
        pred_c, pred_s, engine, is_escalated = _normalize(
            label_fn, e["text"], e.get("source", "社区"),
        )

        engines[engine] += 1
        if is_escalated:
            escalated += 1
        if engine == "small":
            covered += 1
            if pred_c == gold_c:
                covered_l1 += 1
                if pred_s == gold_s:
                    covered_l2 += 1

        conf[gold_c][pred_c] += 1
        l1_flags.append(int(pred_c == gold_c))
        l2_flags.append(int(pred_c == gold_c and pred_s == gold_s))
        if pred_c == gold_c:
            l1 += 1
            if pred_s == gold_s:
                l2 += 1
            else:
                mis.append({"text": e["text"][:40], "gold": f"{gold_c}/{gold_s}",
                            "pred": f"{pred_c}/{pred_s}", "level": "sub"})
        else:
            mis.append({"text": e["text"][:40], "gold": f"{gold_c}/{gold_s}",
                        "pred": f"{pred_c}/{pred_s}", "level": "category"})

    n = len(examples)
    per: dict[str, dict] = {}
    for c in cats:
        tp = conf[c][c]
        fn = sum(conf[c].values()) - tp
        fp = sum(conf[o][c] for o in cats if o != c)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per[c] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
    macro_f1 = sum(m["f1"] for m in per.values()) / len(per) if per else 0.0

    return {
        "total": n,
        "level1_accuracy": round(l1 / n, 4) if n else 0.0,
        "level2_accuracy": round(l2 / n, 4) if n else 0.0,
        "level1_accuracy_ci95": _bootstrap_ci(l1_flags),
        "level2_accuracy_ci95": _bootstrap_ci(l2_flags),
        "macro_f1": round(macro_f1, 4),
        "per_category": per,
        "confusion": {k: dict(v) for k, v in conf.items()},
        "coverage": round(covered / n, 4) if n else 0.0,
        "escalation_rate": round(escalated / n, 4) if n else 0.0,
        "precision_at_coverage_l1": round(covered_l1 / covered, 4) if covered else 0.0,
        "precision_at_coverage_l2": round(covered_l2 / covered, 4) if covered else 0.0,
        "engine_breakdown": dict(engines),
        "misclassified": mis,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="VOC 分类器评估")
    ap.add_argument("--mode", choices=["rules", "semantic", "cascade"], default="rules",
                    help="rules=关键词规则 / semantic=LLM / cascade=小模型优先+LLM兜底")
    ap.add_argument("--dataset", type=str, default=None,
                    help="指定数据集文件(JSONL/JSON/CSV/TSV)，默认 data/eval_set.json")
    ap.add_argument("--cascade-fallback", choices=["llm", "rules"], default=None,
                    help="级联低置信兜底通道，默认读配置")
    ap.add_argument("--text-field", default=None, help="文本列名(默认自动识别)")
    ap.add_argument("--category-field", default=None, help="一级分类列名")
    ap.add_argument("--sub-category-field", default=None, help="二级分类列名")
    ap.add_argument("--source-field", default=None, help="来源列名")
    args = ap.parse_args()

    from .config import get_settings
    cfg = get_settings()
    # --mode 是显式评测意图，必须同步到 classify_auto 的分发开关，
    # 否则默认配置(classifier_mode)会让 --mode cascade 静默退化成全规则评测
    cfg.classifier_mode = args.mode
    if args.cascade_fallback:
        cfg.cascade_fallback = args.cascade_fallback
    cfg.enable_pending_pool = False  # 离线评估不写入待标注池

    examples = load_dataset(
        args.dataset, args.text_field, args.category_field,
        args.sub_category_field, args.source_field,
    ) if args.dataset else load_eval_set()
    fn = {"rules": label_rules, "semantic": label_semantic, "cascade": label_cascade}[args.mode]
    print(json.dumps(evaluate(fn, examples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
