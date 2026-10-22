"""
真实用户反馈采集 —— 从 Apple App Store 中国区评论接口拉取小米系 App 的真实评论。

数据口径:
  来源   App Store 用户评论(公开接口，无需登录)
  App    小米商城 / 米家 / 小米有品 / 小米汽车 / 小米运动健康 / 小米音箱
  字段   app, app_id, review_id, rating(1-5), title, content, date, source

用法:
    python -m scripts.fetch_real_reviews --pages 10
    python -m scripts.fetch_real_reviews --pages 3 --sample 300

说明:
  - 接口按页返回，每页约 50 条；页数越多越可能拿到历史评论，重复 ID 自动去重。
  - 采样按「App × 星级」分层，保证低分(抱怨)与高分(好评)都有覆盖。
  - 该脚本只落原始数据，标签由人工标注脚本 consume_real_labels.py 生成。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "real"

APPS = {
    "小米商城": (1024958046, "商城"),
    "米家": (957323480, "商城"),
    "小米有品": (1212182562, "商城"),
    "小米汽车": (6473663234, "商城"),
    "小米运动健康": (1493500777, "服务频道"),
    "小米音箱": (1270594280, "服务频道"),
}

_FEED = "https://itunes.apple.com/cn/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json"


def _get(url: str, timeout: int = 25) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def _entries(app: str, app_id: int, page: int, source: str) -> list[dict]:
    try:
        feed = _get(_FEED.format(page=page, app_id=app_id))
    except Exception as e:  # 单页失败不影响整体采集
        print(f"  [warn] {app} page={page} 拉取失败: {e!r}")
        return []

    rows = []
    for e in feed.get("feed", {}).get("entry", []):
        rating = e.get("im:rating", {}).get("label")
        content = (e.get("content", {}).get("label") or "").strip()
        title = (e.get("title", {}).get("label") or "").strip()
        if not rating or not content:  # 首条为 App 元信息，跳过
            continue
        rid = e.get("id", {}).get("label") or hashlib.md5(content.encode()).hexdigest()
        rows.append({
            "app": app, "app_id": app_id, "review_id": rid,
            "rating": int(rating), "title": title, "content": content,
            "date": (e.get("updated", {}).get("label") or "")[:10],
            "source": source,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="拉取 App Store 真实评论")
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--sample", type=int, default=300, help="抽样条数(按App×星级分层)")
    ap.add_argument("--out", default=str(DATA_DIR / "reviews_raw.jsonl"))
    ap.add_argument("--queue", default=str(DATA_DIR / "label_queue.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict] = []

    for app, (app_id, source) in APPS.items():
        before = len(rows)
        for page in range(1, args.pages + 1):
            for r in _entries(app, app_id, page, source):
                key = hashlib.md5(f"{r['content']}".encode()).hexdigest()
                if key in seen or len(r["content"]) < 6:
                    continue
                seen.add(key)
                rows.append(r)
            time.sleep(0.5)  # 轻量限速，避免触发风控
        print(f"{app}: +{len(rows) - before} 条")

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"原始评论 {len(rows)} 条 -> {out}")

    # 按 App × 星级分层抽样，保证抱怨/好评都有覆盖
    rng = random.Random(args.seed)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["app"], r["rating"])].append(r)
    per_bucket = max(1, args.sample // max(1, len(buckets)))
    queue: list[dict] = []
    for key in sorted(buckets):
        pool = buckets[key]
        rng.shuffle(pool)
        queue.extend(pool[:per_bucket])
    rng.shuffle(queue)
    queue = queue[: args.sample]

    qpath = Path(args.queue)
    qpath.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in queue), encoding="utf-8")
    print(f"待标注样本 {len(queue)} 条 -> {qpath}")


if __name__ == "__main__":
    main()
