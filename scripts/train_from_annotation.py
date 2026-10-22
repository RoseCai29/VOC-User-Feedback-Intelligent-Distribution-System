"""
基于新标注数据训练 BERT 小模型。

数据可以是 xlsx / csv / tsv / jsonl，分类列优先取「人工标注一级分类 / 人工标注二级分类」，
为空时回退「参考一级分类 / 参考二级分类」。训练前会做标签校验、文本去重和分布统计。

用法:
    python -m scripts.train_from_annotation --input data/标注数据.xlsx

    # 只检查数据质量，不训练
    python -m scripts.train_from_annotation --input data/标注数据.xlsx --dry-run

    # 正式训练(CPU 友好主干；追求精度可用 bert-base-chinese)
    $env:HF_ENDPOINT="https://hf-mirror.com"
    python -m scripts.train_from_annotation --input data/标注数据.xlsx \
        --output data/models/bert_voc --epochs 3 --batch-size 48
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.classifier.annotations import load_annotations
from src.classifier.bert import train_bert
from src.state import FeedbackSubCategory


def main() -> None:
    ap = argparse.ArgumentParser(description="基于标注表训练 VOC BERT 小模型")
    ap.add_argument("--input", required=True, help="标注数据(xlsx/csv/tsv/jsonl)")
    ap.add_argument("--output", default="data/models/bert_voc", help="模型输出目录")
    ap.add_argument("--backbone", default="hfl/rbt3", help="预训练主干")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-field", default=None, help="文本列名(默认自动识别)")
    ap.add_argument("--source-field", default=None, help="来源列名(默认自动识别)")
    ap.add_argument("--min-per-class", type=int, default=20,
                    help="每个二级类的最低样本数，低于该值仅告警")
    ap.add_argument("--dry-run", action="store_true", help="只做数据校验与统计")
    args = ap.parse_args()

    examples, stats = load_annotations(args.input, args.text_field, args.source_field)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if stats["used"] == 0:
        raise SystemExit("没有可用样本，请检查文本列与分类列是否为中文标签")

    missing = {
        sub.value: stats["per_sub_category"].get(sub.value, 0)
        for sub in FeedbackSubCategory
        if stats["per_sub_category"].get(sub.value, 0) < args.min_per_class
    }
    if missing:
        print(f"[告警] 以下二级类样本不足 {args.min_per_class} 条，训练效果可能受限: {missing}",
              file=sys.stderr)

    if args.dry_run:
        print("[dry-run] 数据校验完成，未启动训练")
        return

    groups = [e.get("group") or "" for e in examples]
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
        # 没有分组列时传 None，走分层切分，避免单组切分报错
        groups=groups if any(groups) else None,
    )
    report["data"] = str(args.input)
    report["data_stats"] = stats

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
