"""
冷启动伪标签 CLI —— 用大模型给未标注反馈打标，产出小模型训练集。

用法:
  # 1. 未标注数据 → LLM 伪标签训练集(仅保留置信度达标的 LLM 结果)
  python -m src.classifier.bootstrap --input data/raw_feedback.csv --output data/pseudo_labeled.jsonl

  # 2. 导出低置信待标注池，供人工修正
  python -m src.classifier.bootstrap --from-pending --output data/pending_export.jsonl

冷启动阶段小模型缺失时，线上链路同样走 LLM 分类，所以两类样本口径一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import get_settings
from ..tools import classify_semantic
from .dataset import load_examples


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pseudo_label(
    examples: list[dict],
    min_confidence: float = 0.8,
    classifier=classify_semantic,
) -> tuple[list[dict], dict]:
    """逐条调用 LLM 打标，仅保留置信度达标且确由 LLM 产出的样本。"""
    seen: set[str] = set()
    rows: list[dict] = []
    stats = {
        "input_records": len(examples),
        "duplicate_texts": 0,
        "llm_labeled": 0,
        "kept": 0,
        "dropped_low_confidence": 0,
        "dropped_non_llm": 0,
        "dropped_invalid": 0,
    }
    for ex in examples:
        text = ex["text"].strip()
        if not text or text in seen:
            if text:
                stats["duplicate_texts"] += 1
            continue
        seen.add(text)
        try:
            result = json.loads(classifier(text, ex.get("source", "社区")))
        except Exception:
            stats["dropped_invalid"] += 1
            continue

        engine = result.get("engine", "")
        confidence = float(result.get("confidence") or 0.0)
        if engine != "llm":
            stats["dropped_non_llm"] += 1
            continue
        stats["llm_labeled"] += 1
        if confidence < min_confidence:
            stats["dropped_low_confidence"] += 1
            continue
        rows.append({
            "text": text,
            "category": result.get("category", ""),
            "sub_category": result.get("sub_category", ""),
            "confidence": confidence,
            "engine": engine,
            "source": ex.get("source", "社区"),
            "label_type": "pseudo",
        })
        stats["kept"] += 1
    return rows, stats


def export_pending(path: Path, status: str = "pending", limit: int = 5000) -> dict:
    """导出待标注池，供人工修正后再回流训练。"""
    from ..db import list_pending_labels

    records = list_pending_labels(status=status or None, limit=limit)
    rows = [{
        "text_hash": r.text_hash,
        "text": r.raw_text,
        "source": r.source,
        "product_line": r.product_line,
        "small_category": r.small_category,
        "small_sub_category": r.small_sub_category,
        "small_category_confidence": r.small_category_confidence,
        "small_sub_confidence": r.small_sub_confidence,
        "llm_category": r.llm_category,
        "llm_sub_category": r.llm_sub_category,
        "llm_confidence": r.llm_confidence,
        "llm_engine": r.llm_engine,
        "status": r.status,
        "human_category": r.human_category,
        "human_sub_category": r.human_sub_category,
    } for r in records]
    _write_jsonl(path, rows)
    return {"status": status, "exported": len(rows), "output": str(path)}


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser(description="VOC 小模型冷启动: LLM 伪标签 / 待标注池导出")
    ap.add_argument("--input", help="未标注反馈文件(JSONL/JSON/CSV/TSV)")
    ap.add_argument("--output", required=True, help="输出 JSONL 路径")
    ap.add_argument("--min-confidence", type=float, default=cfg.bootstrap_min_confidence,
                    help=f"伪标签保留的最低 LLM 置信度(默认 {cfg.bootstrap_min_confidence})")
    ap.add_argument("--limit", type=int, default=5000, help="单次最多处理条数")
    ap.add_argument("--from-pending", action="store_true", help="从待标注池导出而非调用 LLM")
    ap.add_argument("--status", default="pending", help="--from-pending 时的状态过滤(pending/approved/rejected/空=全部)")
    ap.add_argument("--text-field", default=None, help="文本列名(默认自动识别)")
    ap.add_argument("--source-field", default=None, help="来源列名")
    args = ap.parse_args()

    output = Path(args.output)
    if args.from_pending:
        report = export_pending(output, args.status, args.limit)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if not args.input:
        raise SystemExit("--input 与 --from-pending 必须提供其一")

    examples = load_examples(args.input, text_field=args.text_field,
                             source_field=args.source_field)[:args.limit]
    rows, stats = pseudo_label(examples, args.min_confidence)
    _write_jsonl(output, rows)
    stats["min_confidence"] = args.min_confidence
    stats["output"] = str(output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
