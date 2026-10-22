"""
BERT 小模型训练 CLI。

用法:
  python -m src.classifier.train_bert --data data/voc_dataset.jsonl --output data/models/bert_voc

国内网络可先设置镜像:
  $env:HF_ENDPOINT="https://hf-mirror.com"
"""

from __future__ import annotations

import argparse
import json

from ..state import FeedbackCategory, FeedbackSubCategory
from .bert import train_bert
from .dataset import load_examples

_VALID_CATEGORIES = {c.value for c in FeedbackCategory}
_VALID_SUBS = {s.value for s in FeedbackSubCategory}


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 VOC BERT 小模型")
    ap.add_argument("--data", default="data/voc_dataset.jsonl", help="标注数据集(JSONL/JSON/CSV/TSV)")
    ap.add_argument("--output", default="data/models/bert_voc", help="模型输出目录")
    ap.add_argument("--backbone", default="hfl/rbt3",
                    help="预训练主干(rbt3 为 CPU 友好版；追求精度可用 bert-base-chinese)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-field", default=None, help="文本列名(默认自动识别)")
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
    if len(examples) < 30:
        raise SystemExit(f"有效训练样本不足: {len(examples)}（原始 {len(raw)} 条，剔除标签非法 {skipped} 条）")

    report = train_bert(
        [e["text"] for e in examples],
        [e["category"] for e in examples],
        [e["sub_category"] for e in examples],
        output=args.output,
        backbone_name=args.backbone,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        test_size=args.test_size,
        seed=args.seed,
        groups=[e["template_id"] for e in examples] if all(e.get("template_id") for e in examples) else None,
    )
    report.update({
        "data": str(args.data),
        "raw_examples": len(raw),
        "dropped_invalid_labels": skipped,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
