"""
把商城评论的星级弱监督转为四分类训练样本，并与现有训练集合并。

星级只提供粗粒度极性（4~5 星=正向，1~2 星=负面），负面样本再用规则引擎细分；
3 星视为中性直接丢弃。输出的每条样本都带 label_source，便于区分人工/弱监督标签。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.classifier.dataset import read_records
from src.tools import _detect_product_line, classify

ID_FIELDS = ("评论ID", "id", "review_id")
TITLE_FIELDS = ("评论标题", "标题", "title")
CONTENT_FIELDS = ("评论内容", "内容", "content", "评论")
RATING_FIELDS = ("评分（1-5）", "评分(1-5)", "评分", "rating", "star")

SERVICE_CUES = ("客服", "售后", "物流", "快递", "配送", "服务", "发货", "安装", "退换")


def _value(record: dict, fields: tuple[str, ...]) -> str:
    for key in fields:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _rating(record: dict) -> int | None:
    raw = _value(record, RATING_FIELDS)
    match = re.search(r"[1-5]", raw)
    return int(match.group()) if match else None


def weak_label(text: str, rating: int) -> tuple[str, str, str, float] | None:
    """返回 (category, sub_category, label_source, confidence)；3 星返回 None。"""
    if rating >= 4:
        sub = "服务好评" if any(c in text for c in SERVICE_CUES) else "产品好评"
        return "正向反馈", sub, "rating", 0.9
    if rating <= 2:
        out = json.loads(classify(text, "商城"))
        if out["category"] == "正向反馈" or out["confidence"] < 0.55:
            # 规则没识别出具体问题时，按泛化质量吐槽处理，并标记为低置信弱标签
            return "舆情吐槽", "产品质量", "rating+fallback", 0.5
        return out["category"], out["sub_category"], "rating+rule", out["confidence"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="生成四分类训练集（商城评论 + 现有数据）")
    parser.add_argument("--reviews", required=True, help="商城评论 xlsx/csv")
    parser.add_argument("--base", default="data/voc_dataset.jsonl", help="现有训练集")
    parser.add_argument("--output", default="data/voc_train_v2.jsonl")
    args = parser.parse_args()

    examples: list[dict] = []
    seen: set[str] = set()
    stats = Counter()

    base_path = Path(args.base)
    if base_path.exists():
        for line in base_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            text = re.sub(r"\s+", " ", row["text"]).strip()
            if text in seen:
                stats["base_duplicate"] += 1
                continue
            seen.add(text)
            examples.append({
                "text": text,
                "source": row.get("source", "社区"),
                "product_line": row.get("product_line", "其他"),
                "category": row["category"],
                "sub_category": row["sub_category"],
                "group": row.get("template_id", ""),
                "label_source": "base",
            })
            stats["base_used"] += 1

    for record in read_records(args.reviews):
        title = _value(record, TITLE_FIELDS)
        content = _value(record, CONTENT_FIELDS)
        text = re.sub(r"\s+", " ", f"{title} {content}").strip()
        rating = _rating(record)
        if not text or rating is None:
            stats["mall_dropped_missing"] += 1
            continue
        label = weak_label(text, rating)
        if label is None:
            stats["mall_skipped_neutral"] += 1
            continue
        category, sub, source, confidence = label
        if text in seen:
            stats["mall_duplicate"] += 1
            continue
        seen.add(text)
        examples.append({
            "text": text,
            "source": "商城",
            "product_line": _detect_product_line(text).value,
            "category": category,
            "sub_category": sub,
            "group": f"mall:{_value(record, ID_FIELDS)}",
            "label_source": source,
            "weak_confidence": confidence,
            "rating": rating,
        })
        stats[f"mall_{source}"] += 1
        stats[f"rating_{rating}"] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cats = Counter(e["category"] for e in examples)
    subs = Counter(f"{e['category']}/{e['sub_category']}" for e in examples)
    print(json.dumps({
        "output": str(output),
        "total": len(examples),
        "stats": dict(stats),
        "categories": dict(cats),
        "sub_categories": dict(sorted(subs.items())),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
