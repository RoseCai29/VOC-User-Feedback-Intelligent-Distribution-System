"""
VOC 智能分发系统 —— 6 个 Function Calling 工具
全注册为 OpenAI-compatible tool schema，LLM 自主决定调用次序。

含金量躍升点:
  retrieve_evidence → ChromaDB 向量语义检索 (非玩具 Jaccard)
  crawl_feedback     → httpx 异步爬虫骨架 (非硬编码 mock)
"""

from __future__ import annotations

import json
import os
import hashlib
import logging
import threading

from .state import (
    ClassificationResult,
    DedupBucket, RouteTarget, ServiceGuide,
    FeedbackCategory, FeedbackSubCategory,
    XiaomiProductLine,
)
from .config import get_settings
from .db import get_all_dedup_buckets, upsert_dedup_bucket, clear_dedup_buckets

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════

# 历史工单 (ChromaDB 种子数据)

# ═══════════════════════════════════════════

_HISTORICAL_TICKETS: list[dict] = [
    {"id": "T001", "content": "APP闪退严重，打开作品页面就崩溃，售后保修需要解决", "category": "售后报修", "sub": "软件缺陷", "resolution": "已修复v2.3.1"},
    {"id": "T002", "content": "建议增加夜间模式，晚上看屏幕太刺眼了", "category": "作品建议", "sub": "界面优化", "resolution": "已排期Q4"},
    {"id": "T003", "content": "物流太慢了，等了5天还没发货", "category": "售后报修", "sub": "物流问题", "resolution": "已催促仓库"},
    {"id": "T004", "content": "希望能支持视频倍速播放功能", "category": "作品建议", "sub": "功能建议", "resolution": "已加入需求池"},
    {"id": "T005", "content": "客服态度极差，打电话投诉也没人处理", "category": "舆情吐槽", "sub": "服务态度", "resolution": "已回访道歉"},
    {"id": "T006", "content": "产品质量太差，用了三天就坏了", "category": "舆情吐槽", "sub": "产品质量", "resolution": "已安排退换"},
    {"id": "T007", "content": "价格比其他平台贵好多，不值这个价", "category": "舆情吐槽", "sub": "价格问题", "resolution": "已解释定价策略"},
    {"id": "T008", "content": "设备无法开机，充电没反应需要售后维修", "category": "售后报修", "sub": "硬件故障", "resolution": "已派单维修"},
    {"id": "T009", "content": "推荐算法不准确，总是推不感兴趣的内容", "category": "作品建议", "sub": "内容推荐", "resolution": "已优化算法v3"},
    {"id": "T010", "content": "APP经常闪退崩溃，打开就黑屏需要保修处理", "category": "售后报修", "sub": "软件缺陷", "resolution": "修复中"},
    {"id": "T011", "content": "小米14 Ultra夜景模式偏黄，算法需要优化", "category": "作品建议", "sub": "功能建议", "resolution": "已反馈影像组"},
    {"id": "T012", "content": "Redmi K70充电发热严重，用了三个月充不进电了", "category": "售后报修", "sub": "硬件故障", "resolution": "已换新"},
    {"id": "T013", "content": "小米电视开机广告太长，用户体验很差", "category": "舆情吐槽", "sub": "产品质量", "resolution": "已调整广告策略"},
    {"id": "T014", "content": "SU7交付等8个月，产能不足影响体验", "category": "舆情吐槽", "sub": "服务态度", "resolution": "已加急排产"},
    {"id": "T015", "content": "智能门锁指纹识别经常失败，安全性存疑", "category": "售后报修", "sub": "硬件故障", "resolution": "已推送固件更新"},
]

_dedup_lock = threading.RLock()
_dedup_cache: dict[str, DedupBucket] = {}
_dedup_cache_loaded = False


def reset_dedup_buckets():
    """清空去重桶（内存缓存 + 持久化存储），供测试/重置使用。"""
    global _dedup_cache, _dedup_cache_loaded
    clear_dedup_buckets()
    _dedup_cache = {}
    _dedup_cache_loaded = True

# ═══════════════════════════════════════════

# ChromaDB 证据检索引擎

# ═══════════════════════════════════════════

_chroma_collection = None

def _get_chroma_collection():
    """懒加载 ChromaDB collection，不可用时返回 None 降级 Jaccard"""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if os.environ.get("SKIP_CHROMADB") == "1":
        logger.info("SKIP_CHROMADB=1，跳过 ChromaDB，直接降级 Jaccard")
        _chroma_collection = None
        return None
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        cfg = get_settings()
        client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
        try:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=cfg.openai_api_key,
                api_base=cfg.openai_base_url,
                model_name=cfg.embedding_model,
            )
        except Exception:
            ef = embedding_functions.DefaultEmbeddingFunction()
        _chroma_collection = client.get_or_create_collection(
            name=cfg.chroma_collection, embedding_function=ef,
        )
        if _chroma_collection.count() == 0:
            _seed_chroma(_chroma_collection)
        logger.info("ChromaDB 证据检索引擎就绪, %d条", _chroma_collection.count())
        return _chroma_collection
    except Exception as e:
        logger.warning("ChromaDB 不可用，降级为 Jaccard 检索: %s", e)
        _chroma_collection = None
        return None

def _seed_chroma(collection):
    docs = [t["content"] for t in _HISTORICAL_TICKETS]
    ids = [t["id"] for t in _HISTORICAL_TICKETS]
    metadatas = [{"category": t["category"], "sub": t["sub"], "resolution": t["resolution"]} for t in _HISTORICAL_TICKETS]
    collection.add(documents=docs, ids=ids, metadatas=metadatas)

# ═══════════════════════════════════════════

# OpenAI Tool Schemas

# ═══════════════════════════════════════════

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "classify",
            "description": "三级分类(作品建议/舆情吐槽/售后报修)+置信度。confidence<0.7时需调retrieve_evidence取证后重分类。",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_text": {"type": "string", "description": "反馈原文"},
                    "source": {"type": "string", "enum": ["社区", "商城", "服务频道"]},
                    "evidence_context": {"type": "string", "description": "可选: 历史证据摘要，用于低置信重分类纠正"},
                },
                "required": ["raw_text", "source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_evidence",
            "description": "ChromaDB向量语义检索历史工单。classify置信度<0.7时LLM先判断可行性再调用此工具，结果写入evidence_ids后自主重分类。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索query"},
                    "top_k": {"type": "integer", "description": "Top-K，默认3", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dedup",
            "description": "n-gram(2~4) Jaccard去重，相似度>threshold归入已有桶。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待去重文本"},
                    "threshold": {"type": "number", "description": "阈值，默认0.75", "default": 0.75},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route",
            "description": "产品线x分类二维倒排查表确定性路由，零LLM参与零幻觉。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["作品建议", "舆情吐槽", "售后报修"]},
                    "sub_category": {"type": "string"},
                    "source": {"type": "string", "enum": ["社区", "商城", "服务频道"]},
                },
                "required": ["category", "sub_category", "source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "push_service_guide",
            "description": "售后渠道推送: 分类为售后报修时，向用户推送售后电话与自助服务链接，而非生成内部工单。",
            "parameters": {
                "type": "object",
                "properties": {
                    "feedback_text": {"type": "string", "description": "反馈原文"},
                    "sub_category": {"type": "string", "description": "售后二级分类(硬件故障/软件缺陷/物流问题)"},
                    "source": {"type": "string", "enum": ["社区", "商城", "服务频道"]},
                },
                "required": ["feedback_text", "sub_category", "source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crawl_feedback",
            "description": "httpx异步爬取小米社区/商城/服务频道反馈，支持按产品线过滤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "product_line": {"type": "string", "enum": ["手机", "电视", "笔记本", "平板", "智能穿戴", "智能家居", "IoT生态链", "大家电", "路由器", "配件", "小米汽车", "其他", ""]},
                    "source": {"type": "string", "enum": ["社区", "商城", "服务频道", ""]},
                    "max_items": {"type": "integer", "description": "最大爬取条数", "default": 10},
                },
                "required": ["url"],
            },
        },
    },
]

# ═══════════════════════════════════════════

# 倒排路由表

# ═══════════════════════════════════════════

_ROUTE_TABLE: dict[tuple[str, str, str], RouteTarget] = {
    ("社区", "作品建议", "功能建议"):     RouteTarget(product_line="社区产品", sub_line="功能迭代", assignee="张三", sla_hours=72),
    ("社区", "作品建议", "界面优化"):     RouteTarget(product_line="社区产品", sub_line="UED",     assignee="李四", sla_hours=72),
    ("社区", "作品建议", "内容推荐"):     RouteTarget(product_line="社区产品", sub_line="算法组",   assignee="王五", sla_hours=72),
    ("社区", "舆情吐槽", "产品质量"):     RouteTarget(product_line="社区产品", sub_line="品控",     assignee="赵六", sla_hours=24),
    ("社区", "舆情吐槽", "服务态度"):     RouteTarget(product_line="社区产品", sub_line="客服部",   assignee="钱七", sla_hours=24),
    ("社区", "舆情吐槽", "价格问题"):     RouteTarget(product_line="社区产品", sub_line="运营",     assignee="孙八", sla_hours=48),
    ("社区", "售后报修", "硬件故障"):     RouteTarget(product_line="社区产品", sub_line="硬件售后", assignee="周九", sla_hours=12),
    ("社区", "售后报修", "软件缺陷"):     RouteTarget(product_line="社区产品", sub_line="研发一组", assignee="吴十", sla_hours=24),
    ("社区", "售后报修", "物流问题"):     RouteTarget(product_line="社区产品", sub_line="供应链",   assignee="郑一", sla_hours=12),
    ("商城", "作品建议", "功能建议"):     RouteTarget(product_line="商城产品", sub_line="产品组",   assignee="商城张三", sla_hours=72),
    ("商城", "作品建议", "界面优化"):     RouteTarget(product_line="商城产品", sub_line="UED",      assignee="商城李四", sla_hours=72),
    ("商城", "作品建议", "内容推荐"):     RouteTarget(product_line="商城产品", sub_line="推荐组",   assignee="商城王五", sla_hours=72),
    ("商城", "舆情吐槽", "产品质量"):     RouteTarget(product_line="商城产品", sub_line="品控",     assignee="商城赵六", sla_hours=24),
    ("商城", "舆情吐槽", "服务态度"):     RouteTarget(product_line="商城产品", sub_line="客服部",   assignee="商城钱七", sla_hours=24),
    ("商城", "舆情吐槽", "价格问题"):     RouteTarget(product_line="商城产品", sub_line="运营",     assignee="商城孙八", sla_hours=48),
    ("商城", "售后报修", "硬件故障"):     RouteTarget(product_line="商城产品", sub_line="硬件售后", assignee="商城周九", sla_hours=12),
    ("商城", "售后报修", "软件缺陷"):     RouteTarget(product_line="商城产品", sub_line="研发组",   assignee="商城吴十", sla_hours=24),
    ("商城", "售后报修", "物流问题"):     RouteTarget(product_line="商城产品", sub_line="供应链",   assignee="商城郑一", sla_hours=12),
    ("服务频道", "作品建议", "功能建议"): RouteTarget(product_line="服务平台", sub_line="产品组",   assignee="服务张三", sla_hours=72),
    ("服务频道", "作品建议", "界面优化"): RouteTarget(product_line="服务平台", sub_line="UED",      assignee="服务李四", sla_hours=72),
    ("服务频道", "作品建议", "内容推荐"): RouteTarget(product_line="服务平台", sub_line="推荐组",   assignee="服务王五", sla_hours=72),
    ("服务频道", "舆情吐槽", "产品质量"): RouteTarget(product_line="服务平台", sub_line="品控",     assignee="服务赵六", sla_hours=24),
    ("服务频道", "舆情吐槽", "服务态度"): RouteTarget(product_line="服务平台", sub_line="客服部",   assignee="服务钱七", sla_hours=24),
    ("服务频道", "舆情吐槽", "价格问题"): RouteTarget(product_line="服务平台", sub_line="运营",     assignee="服务孙八", sla_hours=48),
    ("服务频道", "售后报修", "硬件故障"): RouteTarget(product_line="服务平台", sub_line="硬件售后", assignee="服务周九", sla_hours=12),
    ("服务频道", "售后报修", "软件缺陷"): RouteTarget(product_line="服务平台", sub_line="研发组",   assignee="服务吴十", sla_hours=24),
    ("服务频道", "售后报修", "物流问题"): RouteTarget(product_line="服务平台", sub_line="供应链",   assignee="服务郑一", sla_hours=12),
}

# ═══════════════════════════════════════════

# 小米产品线关键词库

# ═══════════════════════════════════════════

_XIAOMI_PRODUCT_KEYWORDS: dict[XiaomiProductLine, list[str]] = {
    XiaomiProductLine.PHONE:          ["小米14", "Redmi", "手机", "折叠屏", "Mix", "Civi", "K70", "Note"],
    XiaomiProductLine.TV:             ["电视", "小米电视", "ES Pro", "Redmi电视", "大师"],
    XiaomiProductLine.LAPTOP:         ["笔记本", "RedmiBook", "XiaomiBook", "电脑"],
    XiaomiProductLine.TABLET:         ["平板", "小米平板", "Pad", "RedmiPad"],
    XiaomiProductLine.WEARABLE:       ["手表", "手环", "S3", "Watch", "耳机", "Buds"],
    XiaomiProductLine.SMART_HOME:     ["音箱", "门锁", "摄像头", "开关", "灯泡", "网关", "窗帘"],
    XiaomiProductLine.IOT:            ["空净", "净水器", "扫地机", "吸尘器", "加湿器", "传感器"],
    XiaomiProductLine.HOME_APPLIANCE: ["空调", "洗衣机", "冰箱", "热水器", "洗碗机", "油烟机"],
    XiaomiProductLine.ROUTER:         ["路由器", "Mesh", "WiFi", "AX", "BE"],
    XiaomiProductLine.ACCESSORY:      ["充电器", "数据线", "移动电源", "保护壳", "贴膜"],
    XiaomiProductLine.EV:             ["SU7", "汽车", "小米汽车", "自动驾驶", "充电桩"],
}

_MOCK_CRAWLED: list[dict] = [
    {"text": "小米14 Ultra相机夜景模式偏黄，希望优化算法", "source": "社区", "line": "手机"},
    {"text": "Redmi K70充电发热严重，用了三个月充不进电", "source": "商城", "line": "手机"},
    {"text": "小米电视ES Pro 75寸开机广告太长了", "source": "社区", "line": "电视"},
    {"text": "RedmiBook 16屏幕出现竖线，才买两周需要售后", "source": "商城", "line": "笔记本"},
    {"text": "小米手表S3心率监测不准，跑步时差距很大", "source": "社区", "line": "智能穿戴"},
    {"text": "Buds 5 Pro降噪效果差，地铁上还是很吵", "source": "服务频道", "line": "智能穿戴"},
    {"text": "智能门锁Pro开锁反应慢，指纹识别经常失败", "source": "服务频道", "line": "智能家居"},
    {"text": "米家摄像头云台版频繁掉线，WiFi连接不稳定", "source": "社区", "line": "IoT生态链"},
    {"text": "小米空调巨省电Pro制冷效果差，16度都不凉", "source": "商城", "line": "大家电"},
    {"text": "小米洗衣机洗完后门打不开，显示错误代码E4", "source": "服务频道", "line": "大家电"},
    {"text": "AX9000路由器断流严重，打游戏经常460", "source": "社区", "line": "路由器"},
    {"text": "SU7交付等了8个月还没排到", "source": "社区", "line": "小米汽车"},
    {"text": "SU7 Max自动驾驶高速上突然退出", "source": "服务频道", "line": "小米汽车"},
    {"text": "100W充电头用了半年就坏了，售后说不在保修范围", "source": "商城", "line": "配件"},
    {"text": "小米平板6 Pro触控笔连接不上，重启也没用", "source": "服务频道", "line": "平板"},
]

def _detect_product_line(text: str) -> XiaomiProductLine:
    """按最长命中关键词匹配产品线，避免 'Redmi' 抢先命中 'RedmiBook' 等子串误判。"""
    best_line, best_len = XiaomiProductLine.OTHER, 0
    for line, keywords in _XIAOMI_PRODUCT_KEYWORDS.items():
        for kw in keywords:
            if kw in text and len(kw) > best_len:
                best_line, best_len = line, len(kw)
    return best_line

# ═══════════════════════════════════════════

# classify

# ═══════════════════════════════════════════

def classify(raw_text: str, source: str = "社区", evidence_context: str = "") -> str:
    text_lower = raw_text.lower()
    evidence_lower = evidence_context.lower()

    keyword_rules: list[tuple[list[str], FeedbackCategory, FeedbackSubCategory, float]] = [
        (["闪退", "崩溃", "黑屏", "卡顿", "bug", "报错", "异常", "死机", "重启"], FeedbackCategory.AFTER_SALES, FeedbackSubCategory.SOFTWARE_BUG, 0.85),
        (["无法开机", "充不进", "硬件", "屏幕碎了", "按键", "电池", "指纹", "失灵"], FeedbackCategory.AFTER_SALES, FeedbackSubCategory.HARDWARE_FAULT, 0.85),
        (["物流", "发货", "快递", "配送", "发错", "没收到"], FeedbackCategory.AFTER_SALES, FeedbackSubCategory.LOGISTICS_ISSUE, 0.85),
        (["建议", "希望", "能不能", "增加", "新增", "功能", "支持"], FeedbackCategory.SUGGESTION, FeedbackSubCategory.FEATURE_REQUEST, 0.80),
        (["界面", "ui", "ux", "布局", "设计", "颜色", "夜间", "暗黑", "图标", "字体", "动效", "负一屏", "通知栏", "控制中心", "主题"], FeedbackCategory.SUGGESTION, FeedbackSubCategory.UI_IMPROVEMENT, 0.80),
        (["推荐", "算法", "推送", "不感兴趣", "内容"], FeedbackCategory.SUGGESTION, FeedbackSubCategory.CONTENT_RECOMMEND, 0.80),
        (["态度", "客服", "投诉", "服务差", "不处理"], FeedbackCategory.SENTIMENT, FeedbackSubCategory.SERVICE_ATTITUDE, 0.85),
        (["质量", "坏了", "差", "劣质", "次品", "问题", "粗糙", "瑕疵", "做工"], FeedbackCategory.SENTIMENT, FeedbackSubCategory.PRODUCT_QUALITY, 0.75),
        (["贵", "价格", "不值", "涨价", "降价", "坑", "收费"], FeedbackCategory.SENTIMENT, FeedbackSubCategory.PRICE_CONCERN, 0.85),
    ]

    low_triggers = ["用不了", "不好用", "有问题", "不行了", "出毛病"]
    if any(t in text_lower for t in low_triggers):
        if evidence_context:
            if "软件缺陷" in evidence_lower or "闪退" in evidence_lower or "崩溃" in evidence_lower:
                result = ClassificationResult(category=FeedbackCategory.AFTER_SALES, sub_category=FeedbackSubCategory.SOFTWARE_BUG, tags=["稳定性", "证据纠正"], confidence=0.88)
            elif "硬件" in evidence_lower or "故障" in evidence_lower:
                result = ClassificationResult(category=FeedbackCategory.AFTER_SALES, sub_category=FeedbackSubCategory.HARDWARE_FAULT, tags=["证据纠正"], confidence=0.86)
            elif "质量" in evidence_lower:
                result = ClassificationResult(category=FeedbackCategory.SENTIMENT, sub_category=FeedbackSubCategory.PRODUCT_QUALITY, tags=["舆情", "证据纠正"], confidence=0.82)
            else:
                result = ClassificationResult(category=FeedbackCategory.AFTER_SALES, sub_category=FeedbackSubCategory.SOFTWARE_BUG, tags=["证据纠正"], confidence=0.75)
            return json.dumps(result.model_dump(), ensure_ascii=False)
        result = ClassificationResult(category=FeedbackCategory.AFTER_SALES, sub_category=FeedbackSubCategory.SOFTWARE_BUG, tags=["模糊反馈"], confidence=0.45)
        return json.dumps(result.model_dump(), ensure_ascii=False)

    best_match = (FeedbackCategory.SUGGESTION, FeedbackSubCategory.FEATURE_REQUEST, 0.5)
    for keywords, cat, sub, conf in keyword_rules:
        if any(kw in text_lower for kw in keywords):
            best_match = (cat, sub, conf)
            break
    cat, sub, confidence = best_match
    tags: list[str] = []
    if "闪退" in text_lower or "崩溃" in text_lower: tags.append("稳定性")
    if "售后" in text_lower or "保修" in text_lower: tags.append("保修期内")
    if "建议" in text_lower: tags.append("主动反馈")
    if "投诉" in text_lower: tags.append("投诉升级")
    result = ClassificationResult(category=cat, sub_category=sub, tags=tags, confidence=confidence)
    return json.dumps(result.model_dump(), ensure_ascii=False)

# ═══════════════════════════════════════════

_CATEGORY_VALUES = [c.value for c in FeedbackCategory]
_SUBCATEGORY_VALUES = [s.value for s in FeedbackSubCategory]

_SEMANTIC_CLASSIFY_PROMPT = (
    "你是 VOC 用户反馈分类器，只输出一个 JSON 对象。\n"
    f"一级分类必须从 {_CATEGORY_VALUES} 中选择；\n"
    f"二级分类必须从 {_SUBCATEGORY_VALUES} 中选择。\n"
    "分类口径：\n"
    "- 售后报修：设备/软件故障、售后退换、物流发货等需工单处理的问题；\n"
    "- 舆情吐槽：情绪化抱怨，涉及产品质量、服务态度、价格等体验问题；\n"
    "- 作品建议：建议、需求、功能期望等非故障类反馈。\n"
    '输出 JSON 字段：{"category": "...", "sub_category": "...", "tags": ["..."], "confidence": 0.0~1.0}\n'
    "confidence 表示确定程度，模糊或歧义文本请给 <0.7 的低置信度。"
)


def classify_semantic(raw_text: str, source: str = "社区", evidence_context: str = "") -> str:
    """LLM 语义分类，失败自动降级为关键词规则 (classify)。"""
    from .llm import json_completion

    if not raw_text.strip():
        return classify(raw_text, source, evidence_context)

    user = f"反馈原文: {raw_text}\n来源平台: {source}"
    if evidence_context:
        user += f"\n历史证据摘要: {evidence_context}"
    try:
        data = json_completion(_SEMANTIC_CLASSIFY_PROMPT, user)
        cat = next((c for c in FeedbackCategory if c.value == data.get("category")), None)
        sub = next((s for s in FeedbackSubCategory if s.value == data.get("sub_category")), None)
        if cat is None or sub is None:
            raise ValueError(f"非法的分类结果: {data}")
        try:
            conf = float(data.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        tags = [str(t) for t in data.get("tags", [])] if isinstance(data.get("tags"), list) else []
        result = ClassificationResult(category=cat, sub_category=sub, tags=tags, confidence=conf)
        logger.info("LLM语义分类: %s -> %s/%s (%.2f)", raw_text[:20], cat.value, sub.value, conf)
        return json.dumps(result.model_dump(), ensure_ascii=False)
    except Exception as e:
        logger.warning("LLM 语义分类失败，降级关键词规则: %s", e)
        return classify(raw_text, source, evidence_context)


# retrieve_evidence (ChromaDB + Jaccard 降级)

# ═══════════════════════════════════════════

def retrieve_evidence(query: str, top_k: int = 3) -> str:
    coll = _get_chroma_collection()
    if coll is not None:
        try:
            results = coll.query(query_texts=[query], n_results=top_k)
            evidence = []
            for i in range(len(results["ids"][0])):
                evidence.append({
                    "ticket_id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "similarity": round(1.0 - (results["distances"][0][i] if results["distances"] else 0.0), 4),
                    "category": results["metadatas"][0][i].get("category", "") if results["metadatas"] else "",
                    "sub_category": results["metadatas"][0][i].get("sub", "") if results["metadatas"] else "",
                    "resolution": results["metadatas"][0][i].get("resolution", "") if results["metadatas"] else "",
                })
            logger.info("ChromaDB检索: query='%s' → %d条", query[:30], len(evidence))
            return json.dumps({"evidence": evidence, "engine": "chromadb", "total_candidates": coll.count()}, ensure_ascii=False)
        except Exception as e:
            logger.warning("ChromaDB查询失败，降级Jaccard: %s", e)

    # Jaccard 降级 (与 dedup 采用一致的 char n-gram 口径)
    scored = sorted([(_jaccard_sim(query, t["content"]), t) for t in _HISTORICAL_TICKETS], key=lambda x: x[0], reverse=True)
    evidence = [{"ticket_id": t["id"], "content": t["content"], "similarity": round(s, 4), "category": t["category"], "sub_category": t.get("sub", ""), "resolution": t.get("resolution", "")} for s, t in scored[:top_k]]
    return json.dumps({"evidence": evidence, "engine": "jaccard", "total_candidates": len(_HISTORICAL_TICKETS)}, ensure_ascii=False)

# ═══════════════════════════════════════════

# dedup

# ═══════════════════════════════════════════

def _char_ngrams(text: str, n: int) -> set[str]:
    text = text.replace(" ", "")
    return {text[i:i + n] for i in range(len(text) - n + 1)} if len(text) >= n else {text}

def _jaccard_sim(a: str, b: str) -> float:
    sims = []
    for n in (2, 3, 4):
        ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
        sims.append(len(ga & gb) / len(ga | gb) if (ga | gb) else 0.0)
    return sum(sims) / len(sims)

def _load_dedup_cache() -> None:
    global _dedup_cache, _dedup_cache_loaded
    _dedup_cache = {}
    for rec in get_all_dedup_buckets():
        members = json.loads(rec.members) if rec.members else []
        _dedup_cache[rec.bucket_id] = DedupBucket(
            bucket_id=rec.bucket_id, representative=rec.representative,
            members=members, count=rec.count,
        )
    _dedup_cache_loaded = True


def _persist_bucket(bucket: DedupBucket) -> None:
    upsert_dedup_bucket(
        bucket.bucket_id, bucket.representative, bucket.count,
        json.dumps(bucket.members, ensure_ascii=False),
    )


def dedup(text: str, threshold: float = 0.75) -> str:
    """n-gram Jaccard 去重，桶状态持久化 + 线程锁，跨请求/重启不丢失。"""
    global _dedup_cache
    cfg = get_settings()
    text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    with _dedup_lock:
        if not _dedup_cache_loaded:
            _load_dedup_cache()
        best_sim, best_id = 0.0, None
        for bid, bucket in _dedup_cache.items():
            s = _jaccard_sim(text, bucket.representative)
            if s > best_sim:
                best_sim, best_id = s, bid
        if best_sim >= threshold and best_id:
            b = _dedup_cache[best_id]
            b.members.append(text_hash)
            b.members = b.members[-cfg.dedup_max_bucket_members:]  # 成员有界
            b.count += 1
            _persist_bucket(b)
            return json.dumps({"is_duplicate": True, "bucket_id": best_id, "similarity": round(best_sim, 4), "bucket_count": b.count}, ensure_ascii=False)
        new_b = DedupBucket(bucket_id=f"BUCKET-{text_hash}", representative=text, members=[text_hash], count=1)
        _dedup_cache[new_b.bucket_id] = new_b
        _persist_bucket(new_b)
        return json.dumps({"is_duplicate": False, "bucket_id": new_b.bucket_id, "similarity": round(best_sim, 4), "bucket_count": 1}, ensure_ascii=False)

# ═══════════════════════════════════════════

# route

# ═══════════════════════════════════════════

def route(category: str, sub_category: str, source: str) -> str:
    key = (source, category, sub_category)
    target = _ROUTE_TABLE.get(key) or RouteTarget(product_line="通用平台", sub_line="人工分拣", assignee="值班客服", sla_hours=48)
    return json.dumps(target.model_dump(), ensure_ascii=False)

# ═══════════════════════════════════════════

# push_service_guide

# ═══════════════════════════════════════════

_SERVICE_GUIDE_MAP: dict[str, dict] = {
    "硬件故障": {"phone": "400-100-5678", "link": "https://www.mi.com/service/repair", "message": "已为您推送售后维修热线与维修申请入口"},
    "软件缺陷": {"phone": "400-100-5678", "link": "https://www.mi.com/service/online", "message": "已为您推送在线客服与软件支持入口"},
    "物流问题": {"phone": "400-100-5678", "link": "https://www.mi.com/service/track", "message": "已为您推送物流查询与售后服务电话"},
}
_DEFAULT_GUIDE = {"phone": "400-100-5678", "link": "https://www.mi.com/service", "message": "已为您推送售后服务联系方式"}


def push_service_guide(feedback_text: str, sub_category: str, source: str = "社区") -> str:
    guide = _SERVICE_GUIDE_MAP.get(sub_category, _DEFAULT_GUIDE)
    sg = ServiceGuide(phone=guide["phone"], link=guide["link"], message=guide["message"])
    logger.info("售后渠道推送[%s/<%s>]: 电话=%s 链接=%s", sub_category, source, sg.phone, sg.link)
    return json.dumps({"success": True, "guide": sg.model_dump(), "message": sg.message}, ensure_ascii=False)

# ═══════════════════════════════════════════

# crawl_feedback (httpx 异步爬虫骨架)

# ═══════════════════════════════════════════

def _filter_mock(product_line: str, source: str, max_items: int) -> list[dict]:
    items = []
    for item in _MOCK_CRAWLED:
        if len(items) >= max_items: break
        if product_line and item["line"] != product_line: continue
        if source and item["source"] != source: continue
        items.append(item)
    return items


def _live_crawl(url: str, product_line: str, source: str, max_items: int) -> list[dict]:
    """真实 HTTP 爬取骨架 (httpx + BeautifulSoup)，异常由调用方统一降级。"""
    import asyncio

    import httpx
    from bs4 import BeautifulSoup

    cfg = get_settings()

    async def _fetch() -> str:
        async with httpx.AsyncClient(
            timeout=cfg.crawl_timeout,
            follow_redirects=True,
            headers={"User-Agent": cfg.crawl_user_agent},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    html = asyncio.run(_fetch())
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    items: list[dict] = []
    for tag in soup.select("p, li, h1, h2, h3, .title, .content, article, blockquote"):
        text = tag.get_text(strip=True)
        if len(text) < 8 or text in seen:
            continue
        seen.add(text)
        line = _detect_product_line(text).value
        if product_line and line != product_line:
            continue
        items.append({"raw_text": text, "source": source or "社区", "product_line": line})
        if len(items) >= max_items:
            break
    return items


def crawl_feedback(url: str, product_line: str = "", source: str = "", max_items: int = 10) -> str:
    cfg = get_settings()
    items: list[dict] = []
    live_used = False
    if cfg.enable_live_crawl:
        try:
            items = _live_crawl(url, product_line, source, max_items)
            live_used = True
        except Exception as e:
            logger.warning("实时爬取失败，降级内置样例: %s", e)
    if not items:
        items = [
            {"raw_text": item["text"], "source": item["source"],
             "product_line": _detect_product_line(item["text"]).value}
            for item in _filter_mock(product_line, source, max_items)
        ]
    results = [{**item, "url": f"{url}/item/{hash(item['raw_text']) & 0xFFFF:04x}"} for item in items]
    logger.info("爬取完成: %s → %d条 (engine=%s)", url, len(results), "live" if live_used else "mock")
    return json.dumps({"crawled_count": len(results), "engine": "live" if live_used else "mock", "filters": {"product_line": product_line, "source": source, "max_items": max_items}, "items": results, "summary": f"从 {url} 获取到 {len(results)} 条反馈"}, ensure_ascii=False)

# ═══════════════════════════════════════════

# 工具分发

# ═══════════════════════════════════════════

TOOL_DISPATCHER = {
    "classify": classify,
    "retrieve_evidence": retrieve_evidence,
    "dedup": dedup,
    "route": route,
    "push_service_guide": push_service_guide,
    "crawl_feedback": crawl_feedback,
}

def execute_tool(tool_name: str, arguments: dict) -> str:
    func = TOOL_DISPATCHER.get(tool_name)
    if func is None:
        return json.dumps({"error": f"未知工具: {tool_name}"})
    try:
        return func(**arguments)
    except Exception as e:
        logger.exception("工具执行失败: %s", tool_name)
        return json.dumps({"error": str(e)})