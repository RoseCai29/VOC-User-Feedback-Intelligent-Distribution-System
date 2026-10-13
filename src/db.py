"""
数据库持久化 —— SQLAlchemy ORM
"""

from __future__ import annotations

import datetime
from pathlib import Path

from sqlalchemy import create_engine, String, Float, Integer, Boolean, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


class FeedbackRecord(Base):
    __tablename__ = "feedbacks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(32), default="社区")
    product_line: Mapped[str] = mapped_column(String(32), default="其他")
    category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    sub_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    dedup_bucket_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    route_assignee: Mapped[str | None] = mapped_column(String(64), nullable=True)
    guide_phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    guide_link: Mapped[str | None] = mapped_column(String(255), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    trace_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class DedupBucketRecord(Base):
    """去重桶持久化 —— 跨请求/重启共享，避免内存态竞态与丢失"""
    __tablename__ = "dedup_buckets"

    bucket_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    representative: Mapped[str] = mapped_column(Text, default="")
    count: Mapped[int] = mapped_column(Integer, default=1)
    members: Mapped[str] = mapped_column(Text, default="[]")  # JSON 数组，成员摘要
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


_engine = None
_SessionLocal = None


def init_db():
    global _engine, _SessionLocal
    cfg = get_settings()
    db_path = Path(cfg.database_url.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(cfg.database_url, echo=False, connect_args={"check_same_thread": False} if "sqlite" in cfg.database_url else {})
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)


def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def persist_feedback(fb: FeedbackRecord) -> FeedbackRecord:
    with get_session() as sess:
        sess.merge(fb)
        sess.commit()
        return fb


def get_all_dedup_buckets() -> list[DedupBucketRecord]:
    with get_session() as sess:
        return list(sess.query(DedupBucketRecord).all())


def upsert_dedup_bucket(bucket_id: str, representative: str, count: int, members_json: str) -> None:
    with get_session() as sess:
        rec = sess.get(DedupBucketRecord, bucket_id)
        if rec is None:
            rec = DedupBucketRecord(bucket_id=bucket_id)
            sess.add(rec)
        rec.representative = representative
        rec.count = count
        rec.members = members_json
        sess.commit()


def clear_dedup_buckets() -> None:
    with get_session() as sess:
        sess.query(DedupBucketRecord).delete()
        sess.commit()