"""
分类评估 —— 基于标注数据集计算准确率、宏观 F1 与混淆矩阵。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .tools import classify, classify_semantic

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"


def load_eval_set(path: Path | str = EVAL_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["examples"]


def load_dataset(path: Path | str) -> list[dict]:
    """加载标注数据集：JSONL(每条一行) 或 JSON(list 或 {"examples": [...]})。"""
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["examples"] if isinstance(data, dict) else data


def label_rules(text: str, source: str = "社区") -> tuple[str, str]:
    r = json.loads(classify(text, source))
    return r["category"], r["sub_category"]


def label_semantic(text: str, source: str = "社区") -> tuple[str, str]:
    r = json.loads(classify_semantic(text, source))
    return r["category"], r["sub_category"]


def evaluate(label_fn, examples: list[dict]) -> dict:
    """对标注集计算 level-1/level-2 准确率、宏观 F1 与混淆矩阵。"""
    cats = sorted({e["category"] for e in examples})
    conf: dict[str, defaultdict] = {c: defaultdict(int) for c in cats}
    l1 = l2 = 0
    mis: list[dict] = []
    for e in examples:
        gold_c, gold_s = e["category"], e["sub_category"]
        pred_c, pred_s = label_fn(e["text"], e.get("source", "社区"))
        conf[gold_c][pred_c] += 1
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
        "macro_f1": round(macro_f1, 4),
        "per_category": per,
        "confusion": {k: dict(v) for k, v in conf.items()},
        "misclassified": mis,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="VOC 分类器评估")
    ap.add_argument("--mode", choices=["rules", "semantic"], default="rules",
                    help="rules=关键词规则 / semantic=LLM 语义分类")
    ap.add_argument("--dataset", type=str, default=None,
                    help="指定数据集文件(JSONL/JSON)，默认 data/eval_set.json")
    args = ap.parse_args()
    examples = load_dataset(args.dataset) if args.dataset else load_eval_set()
    fn = label_semantic if args.mode == "semantic" else label_rules
    print(json.dumps(evaluate(fn, examples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()