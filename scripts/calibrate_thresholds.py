"""
阈值标定 CLI —— 在留出集上扫描双阈值，输出推荐值。有模板分组时按组切分，否则随机切分。

用法:
    python -m scripts.calibrate_thresholds --data data/voc_dataset.jsonl --target-precision 0.95

选择规则: 二级精度 >= 目标值时最大化小模型直出覆盖率(coverage)，其次优先更高精度。
留出集与训练使用相同的分组切分(seed/test_size)，因此指标反映对未见模板的泛化。
"""

from __future__ import annotations

import argparse
import json

from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.classifier.bert import BertSmallModel
from src.classifier.dataset import load_examples
from src.config import get_settings
from src.state import FeedbackCategory, FeedbackSubCategory

VALID_CATS = {c.value for c in FeedbackCategory}
VALID_SUBS = {s.value for s in FeedbackSubCategory}


def _scan(preds: list[dict], cat_grid: list[float], sub_grid: list[float],
          target: float) -> tuple[list[dict], dict]:
    rows = []
    for ct in cat_grid:
        for st in sub_grid:
            covered = [p for p in preds if p["cat_conf"] >= ct and p["sub_conf"] >= st]
            if not covered:
                continue
            n = len(covered)
            l1 = sum(p["cat"] == p["gold_c"] for p in covered) / n
            l2 = sum(p["cat"] == p["gold_c"] and p["sub"] == p["gold_s"] for p in covered) / n
            rows.append({
                "category_threshold": round(ct, 2),
                "sub_threshold": round(st, 2),
                "coverage": round(n / len(preds), 4),
                "precision_l1": round(l1, 4),
                "precision_l2": round(l2, 4),
                "covered": n,
            })
    passing = [r for r in rows if r["precision_l2"] >= target]
    best = max(passing, key=lambda r: (r["coverage"], r["precision_l2"])) if passing else None
    return rows, best


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser(description="BERT 小模型双阈值标定")
    ap.add_argument("--data", default="data/voc_dataset.jsonl")
    ap.add_argument("--model", default=cfg.small_model_path)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    examples = [
        e for e in load_examples(args.data)
        if e["category"] in VALID_CATS and e["sub_category"] in VALID_SUBS
    ]
    groups = [e.get("template_id") or "" for e in examples]
    indices = list(range(len(examples)))
    # 没有模板分组时退回随机切分，避免单组被 GroupShuffleSplit 拒绝
    if len(set(groups)) > 1:
        _, te_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed,
        ).split(indices, groups=groups))
        split_desc = f"group(template) test_size={args.test_size} seed={args.seed}"
    else:
        _, te_idx = train_test_split(
            indices, test_size=args.test_size, random_state=args.seed,
        )
        split_desc = f"random test_size={args.test_size} seed={args.seed}"
    holdout = [examples[i] for i in te_idx]

    model = BertSmallModel.load(args.model)
    preds = []
    for e in holdout:
        p = model.predict(e["text"])
        if p is None:
            continue
        preds.append({
            "cat": p.category, "sub": p.sub_category,
            "cat_conf": p.category_confidence, "sub_conf": p.sub_confidence,
            "gold_c": e["category"], "gold_s": e["sub_category"],
        })
    if not preds:
        raise SystemExit("留出集推理全部失败，检查模型路径")

    grid = [round(0.55 + 0.02 * i, 2) for i in range(22)]
    rows, best = _scan(preds, grid, grid, args.target_precision)

    current = next(
        (r for r in rows
         if r["category_threshold"] == round(cfg.small_model_category_threshold, 2)
         and r["sub_threshold"] == round(cfg.small_model_sub_threshold, 2)),
        None,
    )
    out = {
        "holdout_size": len(preds),
        "split": split_desc,
        "target_precision_l2": args.target_precision,
        "current": current,
        "recommended": best,
        "top_candidates": sorted(
            rows, key=lambda r: (r["precision_l2"] >= args.target_precision, r["coverage"], r["precision_l2"]),
            reverse=True,
        )[:args.top],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
