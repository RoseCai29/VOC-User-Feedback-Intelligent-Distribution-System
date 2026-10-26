"""
VOC 反馈状态模型 —— LLM 每一步决策序列化为 thought/action/observation 可审计 trace。
相比纯 ML 不可替代: 全链路 Trace / LLM 自主动态编排工具 / 低置信自纠错闭环
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

class FeedbackSource(str, Enum):
    COMMUNITY = "社区"
    MALL = "商城"
    SERVICE = "服务频道"

class XiaomiProductLine(str, Enum):
    PHONE = "手机"
    TV = "电视"
    LAPTOP = "笔记本"
    TABLET = "平板"
    WEARABLE = "智能穿戴"
    SMART_HOME = "智能家居"
    IOT = "IoT生态链"
    HOME_APPLIANCE = "大家电"
    ROUTER = "路由器"
    ACCESSORY = "配件"
    EV = "小米汽车"
    OTHER = "其他"

class FeedbackCategory(str, Enum):
    SUGGESTION = "作品建议"
    SENTIMENT = "舆情吐槽"
    AFTER_SALES = "售后报修"
    POSITIVE = "正向反馈"

class FeedbackSubCategory(str, Enum):
    FEATURE_REQUEST = "功能建议"
    UI_IMPROVEMENT = "界面优化"
    CONTENT_RECOMMEND = "内容推荐"
    PRODUCT_QUALITY = "产品质量"
    SERVICE_ATTITUDE = "服务态度"
    PRICE_CONCERN = "价格问题"
    HARDWARE_FAULT = "硬件故障"
    SOFTWARE_BUG = "软件缺陷"
    LOGISTICS_ISSUE = "物流问题"
    PRODUCT_PRAISE = "产品好评"
    SERVICE_PRAISE = "服务好评"

class ActionType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"

class TraceStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    thought: str = Field(default="")
    action: str = Field(default="")
    observation: str = Field(default="")
    action_type: ActionType = Field(default=ActionType.ACTION)

class ClassificationResult(BaseModel):
    category: FeedbackCategory = Field(description="一级分类")
    sub_category: FeedbackSubCategory = Field(description="二级分类")
    tags: list[str] = Field(default_factory=list, description="三级标签")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class EvidenceItem(BaseModel):
    ticket_id: str
    content: str
    similarity: float = 0.0
    category: Optional[str] = None
    resolution: Optional[str] = None

class DedupBucket(BaseModel):
    bucket_id: str
    representative: str = ""
    members: list[str] = Field(default_factory=list)
    count: int = 1

class RouteTarget(BaseModel):
    product_line: str = ""
    sub_line: str = ""
    assignee: str = ""
    sla_hours: int = 48

class ServiceGuide(BaseModel):
    phone: str = Field(default="", description="售后电话")
    link: str = Field(default="", description="售后服务链接")
    message: str = Field(default="", description="推送给用户的引导话术")

class CrawledFeedback(BaseModel):
    crawl_id: str = Field(default_factory=lambda: f"CW-{uuid.uuid4().hex[:8].upper()}")
    raw_text: str = ""
    source: FeedbackSource = Field(default=FeedbackSource.COMMUNITY)
    product_line: XiaomiProductLine = Field(default=XiaomiProductLine.OTHER)
    url: str = ""
    crawled_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = Field(default_factory=dict)

class FeedbackState(BaseModel):
    feedback_id: str = Field(default_factory=lambda: f"FB-{uuid.uuid4().hex[:8].upper()}")
    raw_text: str = Field(default="", description="剥离 emoji 后的文本，链路统一口径")
    original_text: str = Field(default="", description="用户原始文本，仅用于审计")
    emoji_count: int = Field(default=0)
    emoji_strong_negative: int = Field(default=0)
    emoji_positive: int = Field(default=0)
    emoji_ambiguous: int = Field(default=0, description="语境依赖型表情(😭😅[微笑])，不单独定性")
    emoji_only: bool = Field(default=False, description="原文除表情外无有效内容")
    source: FeedbackSource = Field(default=FeedbackSource.COMMUNITY)
    product_line: XiaomiProductLine = Field(default=XiaomiProductLine.OTHER)

    crawled_feedbacks: list[CrawledFeedback] = Field(default_factory=list)
    crawl_completed: bool = Field(default=False)

    classification: Optional[ClassificationResult] = Field(default=None)

    evidence_ids: list[EvidenceItem] = Field(default_factory=list)
    evidence_retrieved: bool = Field(default=False)

    dedup_bucket: Optional[DedupBucket] = Field(default=None)
    is_duplicate: bool = Field(default=False)

    route_target: Optional[RouteTarget] = Field(default=None)
    service_guide: Optional[ServiceGuide] = Field(default=None)

    trace: list[TraceStep] = Field(default_factory=list)
    # 工具执行失败记录：既用于可观测的降级报告，也用于跳过失败步骤避免无限重试
    errors: list[str] = Field(default_factory=list)
    failed_tools: list[str] = Field(default_factory=list)

    max_retries: int = Field(default=1)
    retry_count: int = Field(default=0)
    finished: bool = Field(default=False)
    next_tool: str = Field(default="")
    next_tool_args: str = Field(default="{}")
    loop_count: int = Field(default=0)
    # LLM 决策的累计 token 消耗，用于离线评估单条成本
    llm_prompt_tokens: int = Field(default=0)
    llm_completion_tokens: int = Field(default=0)
    # 带证据重分类没有带来改进时置位，避免在无信息增量的样本上空转
    reclassify_stalled: bool = Field(default=False)

    use_deterministic: bool = Field(default=True)
    fast_path: bool = Field(default=False, description="小模型高置信直通，未进入 LLM 决策循环")
    confidence_threshold: float = Field(default=0.7)
    dedup_threshold: float = Field(default=0.75)
