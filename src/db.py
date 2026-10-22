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


class PendingLabelRecord(Base):
    """低置信待标注池 —— 升级给 LLM 的样本，供人工修正后回流训练。

    text_hash 为主键，保证同一文本重复投递不产生新行（幂等）。
    """
    __tablename__ = "pending_labels"

    text_hash: Mapped[str] = mapped_column(String(32), primary_key=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(32), default="社区")
    product_line: Mapped[str] = mapped_column(String(32), default="其他")
    small_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    small_sub_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    small_category_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    small_sub_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    llm_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    llm_sub_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    llm_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_engine: Mapped[str | None] = mapped_column(String(16), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | approved | rejected
    human_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    human_sub_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
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


def record_from_state(state, product_line: str) -> FeedbackRecord:
    """把 Agent 终态转成持久化记录（API 与批量入库脚本共用同一映射）。"""
    import json

    cls = state.classification
    rt = state.route_target
    guide = state.service_guide
    bucket = state.dedup_bucket
    return FeedbackRecord(
        id=state.feedback_id, raw_text=state.original_text or state.raw_text, source=state.source.value if hasattr(state.source, "value") else str(state.source),
        product_line=product_line,
        category=cls.category.value if cls else None,
        sub_category=cls.sub_category.value if cls else None,
        confidence=cls.confidence if cls else None,
        tags=json.dumps(cls.tags) if cls and cls.tags else None,
        is_duplicate=state.is_duplicate,
        dedup_bucket_id=bucket.bucket_id if bucket and state.is_duplicate else None,
        route_assignee=rt.assignee if rt else None,
        guide_phone=guide.phone if guide else None,
        guide_link=guide.link if guide else None,
        retry_count=state.retry_count,
        trace_json=json.dumps([
            {"step_id": s.step_id, "thought": s.thought, "action": s.action,
             "observation": s.observation[:100]}
            for s in state.trace
        ], ensure_ascii=False),
    )


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


def add_pending_label(
    text_hash: str,
    raw_text: str,
    source: str = "社区",
    product_line: str = "其他",
    small_category: str | None = None,
    small_sub_category: str | None = None,
    small_category_confidence: float = 0.0,
    small_sub_confidence: float = 0.0,
    llm_category: str | None = None,
    llm_sub_category: str | None = None,
    llm_confidence: float | None = None,
    llm_engine: str | None = None,
) -> bool:
    """写入低置信样本（按 text_hash 幂等）。返回是否新建。

    已有人工标注(status != pending)的记录只更新 LLM 侧信息，不覆盖人工标签。
    """
    with get_session() as sess:
        rec = sess.get(PendingLabelRecord, text_hash)
        created = rec is None
        if rec is None:
            rec = PendingLabelRecord(text_hash=text_hash)
            sess.add(rec)
        rec.raw_text = raw_text
        rec.source = source
        rec.product_line = product_line
        rec.small_category = small_category
        rec.small_sub_category = small_sub_category
        rec.small_category_confidence = small_category_confidence or 0.0
        rec.small_sub_confidence = small_sub_confidence or 0.0
        rec.llm_category = llm_category
        rec.llm_sub_category = llm_sub_category
        rec.llm_confidence = llm_confidence
        rec.llm_engine = llm_engine
        sess.commit()
        return created


def list_pending_labels(status: str | None = None, limit: int = 1000) -> list[PendingLabelRecord]:
    with get_session() as sess:
        q = sess.query(PendingLabelRecord)
        if status:
            q = q.filter(PendingLabelRecord.status == status)
        return list(q.order_by(PendingLabelRecord.created_at).limit(limit).all())


def update_pending_label(
    text_hash: str,
    human_category: str | None = None,
    human_sub_category: str | None = None,
    status: str = "approved",
) -> bool:
    """写回人工标注结果。记录不存在返回 False。"""
    with get_session() as sess:
        rec = sess.get(PendingLabelRecord, text_hash)
        if rec is None:
            return False
        rec.human_category = human_category
        rec.human_sub_category = human_sub_category
        rec.status = status
        sess.commit()
        return True


def count_pending_labels(status: str | None = None) -> int:
    with get_session() as sess:
        q = sess.query(PendingLabelRecord)
        if status:
            q = q.filter(PendingLabelRecord.status == status)
        return q.count()