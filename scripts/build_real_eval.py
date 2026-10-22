"""
真实评估集构建 —— 把人工标注结果与原始评论合并，产出 dev/test 两个集合。

标注来源:
  data/real/label_queue.jsonl   分层抽样的真实 App Store 评论
  data/real/labels_manual.txt   逐条人工标注(索引|一级|二级)

切分方式:
  按 review_id 的 md5 稳定切分，dev≈40% / test≈60%。
  规则只在 dev 上调，test 只用于最终汇报，避免调参污染指标。

用法:
    python -m scripts.build_real_eval
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "real"


def _load_labels(path: Path) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        idx, cat, sub = line.split("|")
        out[int(idx)] = (cat.strip(), sub.strip())
    return out


def main() -> None:
    queue = [json.loads(l) for l in (DATA / "label_queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = _load_labels(DATA / "labels_manual.txt")

    rows = []
    for idx, r in enumerate(queue):
        if idx not in labels:
            continue  # 无法判断意图的样本不入评估集
        cat, sub = labels[idx]
        rows.append({
            "text": r["content"].strip(),
            "title": r["title"],
            "rating": r["rating"],
            "app": r["app"],
            "review_id": r["review_id"],
            "source": r["source"],
            "category": cat,
            "sub_category": sub,
        })

    def bucket(rid: str) -> str:
        return "dev" if int(hashlib.md5(rid.encode()).hexdigest(), 16) % 10 < 4 else "test"

    dev = [r for r in rows if bucket(r["review_id"]) == "dev"]
    test = [r for r in rows if bucket(r["review_id"]) == "test"]

    for name, data in (("eval_real_all", rows), ("eval_real_dev", dev), ("eval_real_test", test)):
        p = DATA / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data), encoding="utf-8")
        dist = Counter(r["sub_category"] for r in data)
        print(f"{name}: {len(data)} 条 -> {p.name} | {dict(sorted(dist.items(), key=lambda x: -x[1]))}")

    print("标注覆盖:", len(rows), "/", len(queue), "，丢弃无明确意图样本", len(queue) - len(rows))


if __name__ == "__main__":
    main()
