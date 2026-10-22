"""
FastAPI 服务 —— VOC 智能分发 REST API
启动: uvicorn src.api:app --host 0.0.0.0 --port 8000
文档: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .config import get_settings
from .logging_config import setup_logging, get_logger
from .graph import run_feedback_invoke
from .state import XiaomiProductLine
from .db import init_db, FeedbackRecord, persist_feedback, get_session, record_from_state

setup_logging()
logger = get_logger(__name__)

_cfg = get_settings()
_cors_origins = [o.strip() for o in _cfg.cors_origins.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("VOC API 启动, 端口=%d", _cfg.server_port)
    yield


app = FastAPI(
    title="VOC 智能分发系统",
    description="基于 LangGraph + LLM 的用户反馈智能分发中枢，支持决策-执行-反思-再决策闭环",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["*"], allow_headers=["*"])

_start_time = time.time()
_feedback_count = 0


class FeedbackRequest(BaseModel):
    raw_text: str = Field(..., description="用户反馈原文", min_length=1, max_length=_cfg.max_input_length)
    source: str = Field("社区", description="来源平台", pattern="^(社区|商城|服务频道)$")
    product_line: str = Field("其他", description="产品线")

    @field_validator("raw_text")
    @classmethod
    def _strip_raw(cls, v: str) -> str:
        """护栏: 去除首尾空白，拒绝纯空白输入。"""
        v = v.strip()
        if not v:
            raise ValueError("反馈原文不能为空")
        return v


class FeedbackResponse(BaseModel):
    feedback_id: str
    category: Optional[str] = None
    sub_category: Optional[str] = None
    confidence: Optional[float] = None
    tags: list[str] = []
    is_duplicate: bool = False
    dedup_bucket_id: Optional[str] = None
    route_assignee: Optional[str] = None
    guide_phone: Optional[str] = None
    guide_link: Optional[str] = None
    retry_count: int = 0
    trace_steps: int = 0
    elapsed_ms: float = 0.0
    mode: str = "agent"
    degraded: bool = False
    errors: list[str] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    uptime_seconds: float = 0.0
    feedbacks_processed: int = 0


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        uptime_seconds=round(time.time() - _start_time, 1),
        feedbacks_processed=_feedback_count,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def process_feedback(req: FeedbackRequest):
    global _feedback_count
    t0 = time.time()
    cfg = get_settings()

    pl = next((l for l in XiaomiProductLine if l.value == req.product_line), XiaomiProductLine.OTHER)

    try:
        # run_feedback_invoke 是同步阻塞的（含 LLM HTTP 调用 + LangGraph 执行），
        # 放入线程池避免阻塞事件循环，保证 /health 等并发请求可用。
        state = await asyncio.to_thread(
            run_feedback_invoke,
            raw_text=req.raw_text, source=req.source,
            use_llm=not cfg.use_deterministic, product_line=pl,
        )
    except Exception as e:
        logger.exception("处理失败")
        raise HTTPException(status_code=500, detail=str(e))

    cls = state.classification
    bucket = state.dedup_bucket
    rt = state.route_target
    guide = state.service_guide

    record = record_from_state(state, req.product_line)
    persist_feedback(record)
    _feedback_count += 1

    return FeedbackResponse(
        feedback_id=state.feedback_id,
        category=cls.category.value if cls else None,
        sub_category=cls.sub_category.value if cls else None,
        confidence=cls.confidence if cls else None,
        tags=cls.tags if cls else [],
        is_duplicate=state.is_duplicate,
        dedup_bucket_id=bucket.bucket_id if bucket and state.is_duplicate else None,
        route_assignee=rt.assignee if rt else None,
        guide_phone=guide.phone if guide else None,
        guide_link=guide.link if guide else None,
        retry_count=state.retry_count,
        trace_steps=len(state.trace),
        elapsed_ms=round((time.time() - t0) * 1000, 1),
        mode=("fast_path" if state.fast_path
              else "llm_agent" if not cfg.use_deterministic else "deterministic"),
        degraded=bool(state.errors),
        errors=state.errors,
    )


@app.get("/feedback/{feedback_id}", response_model=FeedbackResponse)
async def get_feedback(feedback_id: str):
    with get_session() as sess:
        rec = sess.get(FeedbackRecord, feedback_id)
    if not rec:
        raise HTTPException(status_code=404, detail="反馈不存在")
    return FeedbackResponse(
        feedback_id=rec.id, category=rec.category, sub_category=rec.sub_category,
        confidence=rec.confidence,
        tags=json.loads(rec.tags) if rec.tags else [],
        is_duplicate=rec.is_duplicate, dedup_bucket_id=rec.dedup_bucket_id,
        route_assignee=rec.route_assignee, guide_phone=rec.guide_phone,
        guide_link=rec.guide_link, retry_count=rec.retry_count,
    )