"""
小模型训练 CLI。

用法:
  python -m src.classifier.train --data data/voc_dataset.jsonl --output data/models/small_clf.joblib
"""

from __future__ import annotations

import argparse
import json

from sklearn.model_selection import train_test_split

from ..state import FeedbackCategory, FeedbackSubCategory
from .dataset import load_examples
from .linear import LinearSmallModel

_VALID_CATEGORIES = {c.value for c in FeedbackCategory}
_VALID_SUBS = {s.value for s in FeedbackSubCategory}


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 VOC 小模型")
    ap.add_argument("--data", default="data/voc_dataset.jsonl", help="标注数据集(JSONL/JSON/CSV/TSV)")
    ap.add_argument("--output", default="data/models/small_clf.joblib", help="模型输出路径")
    ap.add_argument("--test-size", type=float, default=0.2, help="留出集比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-field", default=None, help="文本列名(默认自动识别 text/raw_text/content...)")
    ap.add_argument("--category-field", default=None, help="一级分类列名")
    ap.add_argument("--sub-category-field", default=None, help="二级分类列名")
    ap.add_argument("--source-field", default=None, help="来源列名")
    args = ap.parse_args()

    raw = load_examples(
        args.data, args.text_field, args.category_field,
        args.sub_category_field, args.source_field,
    )
    examples = [
        e for e in raw
        if e["category"] in _VALID_CATEGORIES and e["sub_category"] in _VALID_SUBS
    ]
    skipped = len(raw) - len(examples)
    if len(examples) < 10:
        raise SystemExit(f"有效训练样本不足: {len(examples)}（原始 {len(raw)} 条，剔除标签非法 {skipped} 条）")

    texts = [e["text"] for e in examples]
    cats = [e["category"] for e in examples]
    subs = [e["sub_category"] for e in examples]

    strat = [f"{c}/{s}" for c, s in zip(cats, subs)]
    try:
        x_tr, x_te, c_tr, c_te, s_tr, s_te = train_test_split(
            texts, cats, subs, test_size=args.test_size, random_state=args.seed, stratify=strat,
        )
    except ValueError:
        # 长尾类别样本过少时退回随机切分
        x_tr, x_te, c_tr, c_te, s_tr, s_te = train_test_split(
            texts, cats, subs, test_size=args.test_size, random_state=args.seed,
        )

    model = LinearSmallModel.train(x_tr, c_tr, s_tr)
    model.save(args.output)

    l1 = l2 = 0
    for text, gold_c, gold_s in zip(x_te, c_te, s_te):
        pred = model.predict(text)
        if pred is None:
            continue
        if pred.category == gold_c:
            l1 += 1
            if pred.sub_category == gold_s:
                l2 += 1

    report = {
        "data": str(args.data),
        "raw_examples": len(raw),
        "dropped_invalid_labels": skipped,
        "train": len(x_tr),
        "test": len(x_te),
        "level1_accuracy": round(l1 / len(x_te), 4),
        "level2_accuracy": round(l2 / len(x_te), 4),
        "classes": {"category": sorted(set(cats)), "sub_category": sorted(set(subs))},
        "model": str(args.output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
