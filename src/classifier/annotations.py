"""
标注表加载 —— 优先使用人工标注列，缺失时回退参考列。

适用于「人工标注一级分类/二级分类 + 参考一级分类/二级分类」这类双列标注表，
也兼容只有单组分类列的数据集。加载时完成别名归一、标签校验、文本去重与分布统计。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..state import FeedbackCategory, FeedbackSubCategory
from .dataset import (
    CATEGORY_ALIASES,
    SOURCE_FIELDS,
    SUB_CATEGORY_ALIASES,
    TEXT_FIELDS,
    read_records,
)

MANUAL_CATEGORY_FIELDS = ("人工标注一级分类", "manual_category", "人工一级分类")
MANUAL_SUB_FIELDS = ("人工标注二级分类", "manual_sub_category", "人工二级分类")
REFERENCE_CATEGORY_FIELDS = ("参考一级分类", "reference_category", "一级分类", "category")
REFERENCE_SUB_FIELDS = ("参考二级分类", "reference_sub_category", "二级分类", "sub_category")

VALID_CATEGORIES = {c.value for c in FeedbackCategory}
VALID_SUBS = {s.value for s in FeedbackSubCategory}
SUB_BY_CATEGORY = {
    "售后报修": {"软件缺陷", "硬件故障", "物流问题"},
    "作品建议": {"功能建议", "界面优化", "内容推荐"},
    "舆情吐槽": {"产品质量", "服务态度", "价格问题"},
    "正向反馈": {"产品好评", "服务好评"},
}


def _first_value(record: dict, candidates: tuple[str, ...]) -> str:
    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _text_of(record: dict, explicit: str | None) -> str:
    keys = (explicit,) if explicit else TEXT_FIELDS
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return re.sub(r"\s+", " ", str(value)).strip()
    return ""


def load_annotations(
    path: str | Path,
    text_field: str | None = None,
    source_field: str | None = None,
) -> tuple[list[dict], dict]:
    """返回 (examples, stats)；examples 字段与训练脚本保持一致。"""
    records = read_records(path)
    stats = {
        "total_records": len(records),
        "used": 0,
        "label_from_manual": 0,
        "label_from_reference": 0,
        "dropped_missing_text": 0,
        "dropped_invalid_label": 0,
        "dropped_inconsistent_pair": 0,
        "dropped_duplicate": 0,
        "unknown_category": Counter(),
        "unknown_sub_category": Counter(),
        "invalid_samples": [],
    }
    examples: list[dict] = []
    seen: set[str] = set()

    for record in records:
        text = _text_of(record, text_field)
        if not text:
            stats["dropped_missing_text"] += 1
            continue
        if text in seen:
            stats["dropped_duplicate"] += 1
            continue

        manual_cat = _first_value(record, MANUAL_CATEGORY_FIELDS)
        manual_sub = _first_value(record, MANUAL_SUB_FIELDS)
        used_manual = bool(manual_cat or manual_sub)
        raw_cat = manual_cat or _first_value(record, REFERENCE_CATEGORY_FIELDS)
        raw_sub = manual_sub or _first_value(record, REFERENCE_SUB_FIELDS)

        category = CATEGORY_ALIASES.get(raw_cat, raw_cat)
        sub_category = SUB_CATEGORY_ALIASES.get(raw_sub, raw_sub)

        invalid_reason = ""
        if category not in VALID_CATEGORIES:
            stats["dropped_invalid_label"] += 1
            stats["unknown_category"][raw_cat or "(空)"] += 1
            invalid_reason = f"一级分类非法: {raw_cat or '(空)'}"
        elif sub_category not in VALID_SUBS:
            stats["dropped_invalid_label"] += 1
            stats["unknown_sub_category"][raw_sub or "(空)"] += 1
            invalid_reason = f"二级分类非法: {raw_sub or '(空)'}"
        elif sub_category not in SUB_BY_CATEGORY[category]:
            stats["dropped_inconsistent_pair"] += 1
            invalid_reason = f"一二级不匹配: {category}/{sub_category}"
        if invalid_reason:
            if len(stats["invalid_samples"]) < 5:
                stats["invalid_samples"].append(
                    {"text": text[:40], "reason": invalid_reason}
                )
            continue

        seen.add(text)
        source = _first_value(record, (source_field,) if source_field else SOURCE_FIELDS) or "社区"
        examples.append({
            "text": text,
            "category": category,
            "sub_category": sub_category,
            "source": source,
            "group": _first_value(record, ("template_id", "group")),
            "label_origin": "manual" if used_manual else "reference",
        })
        stats["used"] += 1
        stats["label_from_manual" if used_manual else "label_from_reference"] += 1

    stats["per_category"] = dict(Counter(e["category"] for e in examples))
    stats["per_sub_category"] = dict(Counter(e["sub_category"] for e in examples))
    stats["per_source"] = dict(Counter(e["source"] for e in examples))
    stats["unknown_category"] = dict(stats["unknown_category"])
    stats["unknown_sub_category"] = dict(stats["unknown_sub_category"])
    return examples, stats
