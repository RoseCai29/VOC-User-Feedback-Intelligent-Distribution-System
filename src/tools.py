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
from .preprocess import strip_emoji
from .db import get_all_dedup_buckets, upsert_dedup_bucket, clear_dedup_buckets, add_pending_label

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
# 初始化失败后置位，避免每次检索都触发一次超时重连（失败一次即永久降级到 Jaccard）
_chroma_disabled = False

def _get_chroma_collection():
    """懒加载 ChromaDB collection，不可用时返回 None 降级 Jaccard"""
    global _chroma_collection, _chroma_disabled
    if _chroma_collection is not None:
        return _chroma_collection
    if _chroma_disabled:
        return None
    if os.environ.get("SKIP_CHROMADB") == "1":
        logger.info("SKIP_CHROMADB=1，跳过 ChromaDB，直接降级 Jaccard")
        _chroma_disabled = True
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
        logger.warning("ChromaDB 不可用，降级为 Jaccard 检索（后续不再重试）: %s", e)
        _chroma_collection = None
        _chroma_disabled = True
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
            "description": "四级分类(作品建议/舆情吐槽/售后报修/正向反馈)+置信度。confidence<0.7时需调retrieve_evidence取证后重分类。",
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
                    "category": {"type": "string", "enum": ["作品建议", "舆情吐槽", "售后报修", "正向反馈"]},
                    "sub_category": {"type": "string"},
                    "source": {"type": "string", "enum": ["社区", "商城", "服务频道"]},
                    "product_line": {"type": "string", "description": "反馈所属产品线，非“其他”时参与路由定责"},
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
    # 正向反馈: 不进工单, 转口碑运营做素材沉淀, SLA 按天级
    ("社区", "正向反馈", "产品好评"):     RouteTarget(product_line="社区产品", sub_line="口碑运营", assignee="社区运营", sla_hours=72),
    ("社区", "正向反馈", "服务好评"):     RouteTarget(product_line="社区产品", sub_line="服务口碑", assignee="社区服务运营", sla_hours=72),
    ("商城", "正向反馈", "产品好评"):     RouteTarget(product_line="商城产品", sub_line="口碑运营", assignee="商城运营", sla_hours=72),
    ("商城", "正向反馈", "服务好评"):     RouteTarget(product_line="商城产品", sub_line="服务口碑", assignee="商城服务运营", sla_hours=72),
    ("服务频道", "正向反馈", "产品好评"): RouteTarget(product_line="服务平台", sub_line="口碑运营", assignee="服务运营", sla_hours=72),
    ("服务频道", "正向反馈", "服务好评"): RouteTarget(product_line="服务平台", sub_line="服务口碑", assignee="服务口碑运营", sla_hours=72),
}

# 产品线责任组：产品线非「其他」时覆盖处置团队，让产品线真正参与定责
_PRODUCT_LINE_OWNER: dict[XiaomiProductLine, str] = {
    XiaomiProductLine.PHONE:          "手机业务组",
    XiaomiProductLine.TV:             "电视业务组",
    XiaomiProductLine.LAPTOP:         "笔记本业务组",
    XiaomiProductLine.TABLET:         "平板业务组",
    XiaomiProductLine.WEARABLE:       "穿戴业务组",
    XiaomiProductLine.SMART_HOME:     "智能家居业务组",
    XiaomiProductLine.IOT:            "生态链业务组",
    XiaomiProductLine.HOME_APPLIANCE: "大家电业务组",
    XiaomiProductLine.ROUTER:         "网络业务组",
    XiaomiProductLine.ACCESSORY:      "配件业务组",
    XiaomiProductLine.EV:             "汽车业务组",
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

def _dump_classification(result: ClassificationResult, engine: str) -> str:
    """统一输出分类结果，并标记实际生效的引擎(便于级联统计)。"""
    payload = result.model_dump()
    payload["engine"] = engine
    return json.dumps(payload, ensure_ascii=False)


def classify(raw_text: str, source: str = "社区", evidence_context: str = "") -> str:
    text_lower = raw_text.lower()
    evidence_lower = evidence_context.lower()

    # 好评优先判定: 只有出现明确称赞、且没有抱怨线索或明确诉求时才归为正向反馈。
    positive_cues = ["满意", "好用", "不错", "喜欢", "推荐", "值得", "回购", "超预期",
                     "好评", "省心", "放心", "给力", "真香", "流畅", "清晰", "舒服",
                     "划算", "性价比", "惊艳", "惊喜", "贴心", "点赞", "棒", "完美",
                     "绝了", "挺好", "方便", "实用", "感谢", "一如既往", "五星",
                     "到货快", "服务好", "智能", "点赞", "很好", "用心", "优秀"]
    complaint_cues = ["不好", "不行", "不满意", "不推荐", "不值", "没用", "没反应",
                      "无法", "不能", "太差", "很差", "太慢", "太贵", "垃圾", "后悔",
                      "失望", "投诉", "故障", "闪退", "崩溃", "报错", "卡顿", "掉线",
                      "断流", "失灵", "发热", "噪音", "广告", "退货", "维修",
                      "但是", "不过", "可惜", "就是", "问题", "延迟", "失败", "错误",
                      "卡", "慢", "崩", "闪退",
                      # 「推荐」既是好评词也是推荐算法场景词，需覆盖其负面表达
                      "不感兴趣", "不准确", "不准", "总推", "乱推"]
    # 明确诉求词：出现时即便有称赞，也应按建议处理，避免「不错，但希望加功能」被判成好评
    request_cues = ["建议", "能不能", "可不可以", "增加", "新增", "支持", "推出",
                    "上线", "适配", "优化", "改进", "修复", "出个", "加个", "求"]
    has_complaint = any(n in text_lower for n in complaint_cues)
    has_request = any(r in text_lower for r in request_cues)
    if any(p in text_lower for p in positive_cues) and not has_complaint and not has_request:
        service_cues = ["客服", "售后", "物流", "快递", "配送", "服务", "发货", "安装"]
        sub = (FeedbackSubCategory.SERVICE_PRAISE if any(s in text_lower for s in service_cues)
               else FeedbackSubCategory.PRODUCT_PRAISE)
        result = ClassificationResult(
            category=FeedbackCategory.POSITIVE, sub_category=sub,
            tags=["正向"], confidence=0.85,
        )
        return _dump_classification(result, "rules")

    # 纯星级好评：真实评论里常见「米家⭐️⭐️⭐️⭐️⭐️👍」这类只给星不给词的表达
    star_count = sum(text_lower.count(s) for s in ("⭐", "★"))
    if star_count >= 3 and not has_complaint and not has_request:
        result = ClassificationResult(
            category=FeedbackCategory.POSITIVE, sub_category=FeedbackSubCategory.PRODUCT_PRAISE,
            tags=["正向"], confidence=0.8,
        )
        return _dump_classification(result, "rules")

    # 规则按「故障严重程度 → 履约 → 价格 → 服务 → 质量 → 界面 → 内容 → 泛诉求」排序。
    # 故障信号优先：真实反馈里「希望能修复闪退」的诉求主体仍是缺陷，不应被建议词吃掉。
    keyword_rules: list[tuple[list[str], FeedbackCategory, FeedbackSubCategory, float]] = [
        # 软件缺陷：崩溃/性能/功能失效/数据同步异常
        (["闪退", "崩溃", "黑屏", "白屏", "卡顿", "卡死", "卡了", "卡的要死", "打不开",
          "加载不出", "加载失败", "加载慢", "加载", "登录不了", "登不上", "登录界面", "闪屏",
          "掉帧", "掉针", "报错", "错误", "异常", "死机", "重启", "无法跳转",
          "点不进去", "自动卸载", "下载失败", "同步不到", "同步失败", "不显示",
          "不通知", "收不到", "没反应", "延迟", "断断续续", "转圈", "付不了款",
          "验证失败", "不生效", "自动填充", "占用", "cpu", "内存", "bug", "闪屏",
          "自动停止", "发热", "烫", "上不了", "用不了", "听不了", "看不了", "玩不了",
          "卡半天", "卡住", "卡成", "无法查询", "影响正常", "数据不准",
          # 真实评论常见表达：功能不可达 / 同步丢数据 / 设备匹配失败
          "看不到", "设置不了", "匹配不上", "连接不上", "不同步", "更新不到",
          "不好使", "特别卡", "很卡", "太卡", "越来越卡", "卡得要命", "无法关闭"],
         FeedbackCategory.AFTER_SALES, FeedbackSubCategory.SOFTWARE_BUG, 0.85),
        # 物流/退换：履约链路阻断
        (["物流", "发货", "快递", "配送", "发错", "没收到", "退货", "退款", "不给退"],
         FeedbackCategory.AFTER_SALES, FeedbackSubCategory.LOGISTICS_ISSUE, 0.85),
        # 硬件故障：设备本体失效
        (["无法开机", "不开机", "充不进", "硬件", "碎了", "坏掉", "坏了", "损坏",
          "不灵敏", "杂音", "续航", "电量", "断连", "连不上", "掉线", "失灵",
          "麦克风", "噪音", "进纸", "指纹", "按键", "门锁打不开", "定位"],
         FeedbackCategory.AFTER_SALES, FeedbackSubCategory.HARDWARE_FAULT, 0.85),
        # 价格/会员/付费（「会员」一词本身不代表价格诉求，需与收费类词共现）
        (["贵", "太贵", "好贵", "要钱", "收费", "付费", "会员费", "涨价", "降价",
          "便宜", "性价比", "不值", "花钱", "坑钱", "买两次"],
         FeedbackCategory.SENTIMENT, FeedbackSubCategory.PRICE_CONCERN, 0.85),
        # 服务态度：客服/售后/门店体验
        (["客服", "售后", "态度", "服务差", "不处理", "投诉", "屏蔽", "不理",
          "没人管", "专员", "答复"],
         FeedbackCategory.SENTIMENT, FeedbackSubCategory.SERVICE_ATTITUDE, 0.85),
        # 产品质量：做工/品控/广告与体验劣化
        (["质量", "垃圾", "劣质", "次品", "粗糙", "瑕疵", "做工", "品控", "广告",
          "弹窗", "杂乱", "臃肿", "占内存", "失望", "拉垮", "无语", "难用",
          "不好用", "不好", "差", "问题"],
         FeedbackCategory.SENTIMENT, FeedbackSubCategory.PRODUCT_QUALITY, 0.75),
        # 界面优化：视觉/布局/适配/无障碍
        (["界面", "ui", "ux", "布局", "排版", "设计", "颜色", "夜间", "深色",
          "暗黑", "图标", "字体", "动效", "负一屏", "通知栏", "控制中心", "主题",
          "全屏", "横屏", "竖屏", "刘海", "点不到", "状态栏", "旁白", "阅读器",
          "看瞎", "显示", "不兼容", "适配"],
         FeedbackCategory.SUGGESTION, FeedbackSubCategory.UI_IMPROVEMENT, 0.80),
        # 内容/推荐/资源丰富度
        (["推荐", "算法", "推送", "不感兴趣", "内容", "曲库", "歌单", "音乐",
          "表盘", "品类", "小说", "资源", "不准确", "总推", "乱推"],
         FeedbackCategory.SUGGESTION, FeedbackSubCategory.CONTENT_RECOMMEND, 0.78),
        # 功能诉求（兜底建议类）
        (["建议", "希望", "能不能", "可不可以", "增加", "新增", "支持", "期待",
          "推出", "上线", "适配", "优化", "改进", "加强", "修复", "出个", "加个", "功能",
          # 真实评论的诉求句式：「为什么没有X」「什么时候能支持」「能出一版」
          "为什么没有", "什么时候", "出一版", "跟进", "没有适配", "应该"],
         FeedbackCategory.SUGGESTION, FeedbackSubCategory.FEATURE_REQUEST, 0.80),
    ]

    # 先跑显式规则，命中即返回
    for keywords, cat, sub, conf in keyword_rules:
        if any(kw in text_lower for kw in keywords):
            tags: list[str] = []
            if "闪退" in text_lower or "崩溃" in text_lower: tags.append("稳定性")
            if "售后" in text_lower or "保修" in text_lower: tags.append("保修期内")
            if "建议" in text_lower: tags.append("主动反馈")
            if "投诉" in text_lower: tags.append("投诉升级")
            result = ClassificationResult(category=cat, sub_category=sub, tags=tags, confidence=conf)
            return _dump_classification(result, "rules")

    # 规则未命中：模糊反馈按「需人工确认」处理，默认归为待研判的吐槽而非建议
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
            return _dump_classification(result, "rules")
        result = ClassificationResult(category=FeedbackCategory.SENTIMENT, sub_category=FeedbackSubCategory.PRODUCT_QUALITY, tags=["模糊反馈"], confidence=0.45)
        return _dump_classification(result, "rules")

    # 兜底：真实反馈里未命中规则的绝大多数是情绪化抱怨，给低置信度交给人工/大模型复核
    result = ClassificationResult(
        category=FeedbackCategory.SENTIMENT, sub_category=FeedbackSubCategory.PRODUCT_QUALITY,
        tags=["未命中规则"], confidence=0.35,
    )
    return _dump_classification(result, "rules")

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
    "- 作品建议：建议、需求、功能期望等非故障类反馈；\n"
    "- 正向反馈：明确表达满意、称赞、推荐意愿，且不含抱怨诉求；\n"
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
    # 小参数模型常把二级类当一级类输出，带错误信息重试一次即可显著降低降级率
    last_err = ""
    for attempt in range(2):
        try:
            prompt_user = user if attempt == 0 else (
                f"{user}\n上次输出不合法({last_err})，请严格只从给定枚举中选择，只输出 JSON。"
            )
            data = json_completion(_SEMANTIC_CLASSIFY_PROMPT, prompt_user)
            cat = next((c for c in FeedbackCategory if c.value == data.get("category")), None)
            sub = next((s for s in FeedbackSubCategory if s.value == data.get("sub_category")), None)
            if cat is None or sub is None:
                raise ValueError(f"category={data.get('category')} sub_category={data.get('sub_category')}")
        except Exception as e:
            last_err = str(e)[:120]
            continue
        try:
            conf = float(data.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        tags = [str(t) for t in data.get("tags", [])] if isinstance(data.get("tags"), list) else []
        result = ClassificationResult(category=cat, sub_category=sub, tags=tags, confidence=conf)
        logger.info("LLM语义分类: %s -> %s/%s (%.2f)", raw_text[:20], cat.value, sub.value, conf)
        return _dump_classification(result, "llm")

    logger.warning("LLM 语义分类失败(重试后仍非法)，降级关键词规则: %s", last_err)
    return classify(raw_text, source, evidence_context)


# ═══════════════════════════════════════════

# 分类通道调度 (小模型优先 + 大模型兜底)

# ═══════════════════════════════════════════

_small_model = None
_small_model_failed = False
_small_model_lock = threading.Lock()


def _get_small_model():
    """懒加载小模型；文件缺失/依赖异常时返回 None，由级联路由降级。"""
    global _small_model, _small_model_failed
    if _small_model is not None or _small_model_failed:
        return _small_model
    with _small_model_lock:
        if _small_model is not None or _small_model_failed:
            return _small_model
        cfg = get_settings()
        try:
            if (cfg.small_model_type or "linear").lower() == "bert":
                from .classifier.bert import BertSmallModel
                _small_model = BertSmallModel.load(cfg.small_model_path)
            else:
                from .classifier.linear import LinearSmallModel
                _small_model = LinearSmallModel.load(cfg.small_linear_model_path)
            logger.info("小模型加载成功: type=%s path=%s", cfg.small_model_type, cfg.small_model_path)
        except Exception as e:
            _small_model_failed = True
            logger.warning("小模型不可用，降级大模型/规则: %s", e)
        return _small_model


def small_model_predict(raw_text: str):
    """对外暴露小模型预测，供 fast path 入口判断是否可整链路直通。"""
    model = _get_small_model()
    if model is None:
        return None
    # emoji 在 rbt3 词表中基本是 [UNK]，只保留文本语义参与分类
    return model.predict(strip_emoji(raw_text))


def _text_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()[:16]


def _capture_pending(result_json: str, raw_text: str, source: str) -> None:
    """级联升级的样本落待标注池；失败只记录日志，不影响主链路。"""
    cfg = get_settings()
    if not cfg.enable_pending_pool:
        return
    try:
        data = json.loads(result_json)
        if not data.get("escalated"):
            return
        sp = data.get("small_prediction") or {}
        add_pending_label(
            text_hash=_text_hash(raw_text),
            raw_text=raw_text,
            source=source,
            small_category=sp.get("category"),
            small_sub_category=sp.get("sub_category"),
            small_category_confidence=sp.get("category_confidence") or 0.0,
            small_sub_confidence=sp.get("sub_confidence") or 0.0,
            llm_category=data.get("category"),
            llm_sub_category=data.get("sub_category"),
            llm_confidence=data.get("confidence"),
            llm_engine=data.get("engine"),
        )
    except Exception as e:
        logger.warning("待标注池写入失败: %s", e)


def classify_auto(raw_text: str, source: str = "社区", evidence_context: str = "") -> str:
    """按配置选择分类通道。

    rules    -> 关键词规则
    semantic -> LLM 语义分类(失败内部降级规则)
    small    -> 纯小模型(无预测则降级规则)
    cascade  -> 小模型高置信直出，低置信升级 LLM/规则(默认)
    """
    cfg = get_settings()
    mode = (cfg.classifier_mode or "cascade").lower()
    # 分类只看文本语义，表情在这里剥离；原始文本仍写入待标注池供人工查看
    text = strip_emoji(raw_text)

    if mode == "rules":
        return classify(text, source, evidence_context)
    if mode == "semantic":
        return classify_semantic(text, source, evidence_context)

    from .classifier.router import CascadeRouter

    model = _get_small_model()
    if mode == "small":
        # 阈值归零：只要有预测就用小模型，无预测时 fallback 到规则
        return CascadeRouter(model, 0.0, 0.0, fallback=classify).classify(
            text, source, evidence_context,
        )

    fallback = classify if (cfg.cascade_fallback or "llm").lower() == "rules" else classify_semantic
    result = CascadeRouter(
        model,
        cfg.small_model_category_threshold,
        cfg.small_model_sub_threshold,
        fallback=fallback,
    ).classify(text, source, evidence_context)
    # 低置信升级样本沉淀，供人工修正后回流训练
    _capture_pending(result, raw_text, source)
    return result


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

def route(category: str, sub_category: str, source: str, product_line: str = "") -> str:
    key = (source, category, sub_category)
    target = _ROUTE_TABLE.get(key) or RouteTarget(product_line="通用平台", sub_line="人工分拣", assignee="值班客服", sla_hours=48)
    # 产品线非「其他」时按业务组定责；来源表继续决定子方向与 SLA
    line = next((l for l in XiaomiProductLine if l.value == product_line), XiaomiProductLine.OTHER)
    owner = _PRODUCT_LINE_OWNER.get(line)
    if owner:
        target = RouteTarget(
            product_line=line.value, sub_line=target.sub_line,
            assignee=f"{owner}/{target.sub_line}", sla_hours=target.sla_hours,
        )
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