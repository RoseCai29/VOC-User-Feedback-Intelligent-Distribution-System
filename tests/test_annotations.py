"""标注表加载测试：人工列优先、参考列回退、标签校验、去重、xlsx 读取。"""

import csv
from pathlib import Path

from openpyxl import Workbook

from src.classifier.annotations import load_annotations


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_manual_labels_take_priority(tmp_path):
    path = _write_csv(tmp_path / "a.csv", [
        {"用户言论": "客服态度很差", "来源": "社区",
         "参考一级分类": "售后报修", "参考二级分类": "硬件故障",
         "人工标注一级分类": "舆情吐槽", "人工标注二级分类": "服务态度"},
        {"用户言论": "物流太慢", "来源": "商城",
         "参考一级分类": "售后报修", "参考二级分类": "物流问题",
         "人工标注一级分类": "", "人工标注二级分类": ""},
    ])
    examples, stats = load_annotations(path)

    assert [e["category"] for e in examples] == ["舆情吐槽", "售后报修"]
    assert [e["sub_category"] for e in examples] == ["服务态度", "物流问题"]
    assert stats["label_from_manual"] == 1
    assert stats["label_from_reference"] == 1
    assert stats["used"] == 2


def test_invalid_and_inconsistent_labels_dropped(tmp_path):
    path = _write_csv(tmp_path / "b.csv", [
        {"用户言论": "分类不存在的反馈", "参考一级分类": "其他", "参考二级分类": "未知"},
        {"用户言论": "一二级不匹配的反馈", "参考一级分类": "售后报修", "参考二级分类": "价格问题"},
        {"用户言论": "", "参考一级分类": "售后报修", "参考二级分类": "软件缺陷"},
        {"用户言论": "正常反馈，需要保修", "参考一级分类": "售后报修", "参考二级分类": "硬件故障"},
    ])
    examples, stats = load_annotations(path)

    assert len(examples) == 1
    assert stats["dropped_missing_text"] == 1
    assert stats["dropped_invalid_label"] == 1
    assert stats["dropped_inconsistent_pair"] == 1
    assert stats["unknown_category"] == {"其他": 1}
    assert len(stats["invalid_samples"]) == 2


def test_alias_normalization_and_dedup(tmp_path):
    path = _write_csv(tmp_path / "c.csv", [
        {"用户言论": "客服不理人", "参考一级分类": "吐槽", "参考二级分类": "客服"},
        {"用户言论": "  客服不理人 ", "参考一级分类": "舆情吐槽", "参考二级分类": "服务态度"},
    ])
    examples, stats = load_annotations(path)

    assert len(examples) == 1
    assert examples[0]["category"] == "舆情吐槽"
    assert examples[0]["sub_category"] == "服务态度"
    assert stats["dropped_duplicate"] == 1


def test_xlsx_with_annotation_columns(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["序号", "用户言论", "来源", "参考一级分类", "参考二级分类",
               "人工标注一级分类", "人工标注二级分类"])
    ws.append([1, "建议增加分屏功能", "社区", "作品建议", "功能建议", "", ""])
    ws.append([2, "屏幕碎了要换屏", "商城", "售后报修", "硬件故障", "售后报修", "硬件故障"])
    path = tmp_path / "d.xlsx"
    wb.save(path)

    examples, stats = load_annotations(path)

    assert stats["total_records"] == 2
    assert stats["used"] == 2
    assert stats["label_from_manual"] == 1
    assert stats["per_category"] == {"作品建议": 1, "售后报修": 1}
    assert examples[0]["text"] == "建议增加分屏功能"
