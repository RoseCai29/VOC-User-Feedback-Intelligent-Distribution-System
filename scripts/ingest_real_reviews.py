"""
真实评论批量入库 —— 把 App Store 真实评论跑完整链路并持久化。

幂等口径:
  feedback_id = "RA-" + md5(source|review_id)[:10]
  同一平台的同一条评论重复投递只会更新同一行，不会新增记录。

用法:
    python -m scripts.ingest_real_reviews                 # 全量入库(rules 模式)
    python -m scripts.ingest_real_reviews --limit 50 --llm
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.config import get_settings
from src.db import FeedbackRecord, get_session, init_db, persist_feedback, record_from_state
from src.graph import run_feedback_invoke
from src.state import XiaomiProductLine

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "real" / "reviews_raw.jsonl"

# App 到产品线的映射，让真实数据也能走到产品线路由
APP_TO_LINE = {
    "小米商城": XiaomiProductLine.PHONE,
    "米家": XiaomiProductLine.SMART_HOME,
    "小米有品": XiaomiProductLine.SMART_HOME,
    "小米汽车": XiaomiProductLine.EV,
    "小米运动健康": XiaomiProductLine.WEARABLE,
    "小米音箱": XiaomiProductLine.IOT,
}


def stable_id(source: str, review_id: str) -> str:
    return "RA-" + hashlib.md5(f"{source}|{review_id}".encode()).hexdigest()[:10]


def main() -> None:
    ap = argparse.ArgumentParser(description="真实评论入库")
    ap.add_argument("--input", default=str(SRC))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    cfg = get_settings()
    cfg.enable_live_crawl = False
    init_db()

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    created = updated = failed = 0
    for r in rows:
        fid = stable_id(r["source"], r["review_id"])
        with get_session() as sess:
            exists = sess.get(FeedbackRecord, fid) is not None
        try:
            state = run_feedback_invoke(
                raw_text=r["content"], source=r["source"], use_llm=args.llm,
                product_line=APP_TO_LINE.get(r["app"], XiaomiProductLine.OTHER),
                feedback_id=fid,
            )
        except Exception as e:
            failed += 1
            print(f"[warn] {fid} 处理失败: {e!r}")
            continue
        persist_feedback(record_from_state(state, APP_TO_LINE.get(r["app"], XiaomiProductLine.OTHER).value))
        created += 0 if exists else 1
        updated += 1 if exists else 0

    with get_session() as sess:
        total = sess.query(FeedbackRecord).count()
    print(f"处理 {len(rows)} 条: 新增 {created} / 已存在更新 {updated} / 失败 {failed}；库内共 {total} 条")


if __name__ == "__main__":
    main()
