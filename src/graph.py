"""
LangGraph 状态机 —— LLM 作为唯一决策中枢

图结构:
  START -> agent_decision (LLM) -> tool_executor -> agent_reflection
              ^                      |                  |
              +--- 再决策(循环) -----+                  |
                                      END <------------+

LLM 通过 OpenAI Function Calling 绑定 6 个工具，每轮自主决定调用哪个、何时收敛。
低置信自纠错: confidence<0.7 -> LLM自主调用 retrieve_evidence -> re-classify (最多3轮)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

from .config import get_settings

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import FeedbackState, TraceStep, ActionType, ClassificationResult
from .state import EvidenceItem, DedupBucket, RouteTarget, ServiceGuide
from .state import FeedbackCategory, FeedbackSubCategory, FeedbackSource
from .state import XiaomiProductLine, CrawledFeedback
from .tools import TOOL_SCHEMAS, execute_tool, classify_semantic
from .compress import build_compressed_context

_SYSTEM_PROMPT = """你是 VOC 用户反馈智能分发中枢。每轮你有权调用一个工具，基于当前状态自主决策。

## 可用工具
| 工具 | 用途 | 何时调用 |
|------|------|----------|
| crawl_feedback | 爬取小米平台反馈 | 尚未爬取时第一步 |
| classify | 三级分类(含置信度) | 有新反馈待分类时 |
| retrieve_evidence | 检索历史相似工单 | 分类置信度<0.7时，先判断可行性再检索 |
| dedup | n-gram Jaccard 去重 | 分类完成后 |
| route | 产品线x分类倒排路由 | 去重完成后 |
| push_service_guide | 推送售后电话/链接 | 分类为"售后报修"时 |

## 低置信自纠错流程 (最多3轮)
1. classify 返回 confidence < 0.7
2. 先判断: 是否有相关历史工单可检索?
3. 若可行 -> 调用 retrieve_evidence(query=反馈原文)
4. 拿到证据后 -> 重新 classify(raw_text=原文, evidence_context=证据摘要)
5. 若置信度仍<0.7且未满3轮 -> 继续; 满3轮 -> 以最终结果收敛

## 收敛条件
- 分类非售后 + 去重完成 + 路由完成 -> FINISHED
- 分类为售后 + 去重 + 路由 + 售后电话/链接推送完成 -> FINISHED
- 置信度低但已满3轮重试 -> FINISHED

## 当前状态上下文
{context}

请用 Function Calling 选择下一步工具，或回复 {{"tool_name":"FINISHED"}} 收敛。"""

def _call_llm(system_prompt: str, tools: list[dict]) -> dict:
    from openai import OpenAI
    cfg = get_settings()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=cfg.llm_timeout,
    )
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请根据当前状态，决定下一步调用哪个工具（仅一个），或回复 FINISHED。"},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=1024,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            return {"thought": f"LLM决策: {tc.function.name}", "tool_name": tc.function.name, "tool_args": args}
        content = (msg.content or "").strip()
        for wrap in ("```json", "```"):
            content = content.replace(wrap, "")
        content = content.strip()
        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            if "FINISHED" in content.upper():
                return {"thought": content, "tool_name": "FINISHED", "tool_args": {}}
            return {"thought": content or "兜底分类", "tool_name": "classify", "tool_args": {}}
    except Exception as e:
        raise RuntimeError(f"LLM 调用失败: {e}") from e

def _deterministic_agent(state: FeedbackState) -> dict:
    """规则引擎降级: LLM 不可用时按固定优先级级联"""
    raw = state.raw_text
    src = state.source.value if hasattr(state.source, 'value') else str(state.source)
    cls = state.classification
    retry = state.retry_count

    if not state.crawl_completed and not raw:
        return {"thought": "降级: 未爬取", "tool_name": "crawl_feedback",
                "tool_args": {"url": "https://www.xiaomi.cn/community/feedbacks", "max_items": 10}}
    if cls is None:
        return {"thought": "降级: 分类", "tool_name": "classify",
                "tool_args": {"raw_text": raw, "source": src}}
    if cls.confidence < state.confidence_threshold and not state.evidence_retrieved and retry < state.max_retries:
        return {"thought": f"降级: 置信度{cls.confidence}<{state.confidence_threshold}，检索证据({retry+1}/{state.max_retries})",
                "tool_name": "retrieve_evidence", "tool_args": {"query": raw, "top_k": 3}}
    if cls.confidence < state.confidence_threshold and state.evidence_retrieved and retry < state.max_retries:
        ev = json.dumps([{"id": e.ticket_id, "content": e.content[:60], "category": e.category}
                         for e in state.evidence_ids], ensure_ascii=False)
        return {"thought": f"降级: 第{retry+1}次重分类", "tool_name": "classify",
                "tool_args": {"raw_text": raw, "source": src, "evidence_context": ev}}
    if state.dedup_bucket is None:
        return {"thought": "降级: 去重", "tool_name": "dedup",
                "tool_args": {"text": raw, "threshold": state.dedup_threshold}}
    if state.route_target is None:
        return {"thought": "降级: 路由", "tool_name": "route",
                "tool_args": {"category": cls.category.value, "sub_category": cls.sub_category.value, "source": src}}
    if cls.category == FeedbackCategory.AFTER_SALES and state.service_guide is None:
        return {"thought": "降级: 推送售后电话/链接", "tool_name": "push_service_guide",
                "tool_args": {"feedback_text": raw, "sub_category": cls.sub_category.value, "source": src}}
    return {"thought": "降级: 完成", "tool_name": "FINISHED", "tool_args": {}}

def _is_complete(state: FeedbackState) -> bool:
    if state.classification is None or state.dedup_bucket is None or state.route_target is None:
        return False
    if state.classification.category == FeedbackCategory.AFTER_SALES and state.service_guide is None:
        return False
    return True


def _bounded_trace(state: FeedbackState, step: TraceStep) -> list[TraceStep]:
    """护栏: trace 轨迹有界，防止长循环/批处理导致内存无限增长。"""
    cfg = get_settings()
    return (list(state.trace) + [step])[-cfg.max_trace_steps:]


def agent_decision_node(state: FeedbackState) -> dict:
    ctx = build_compressed_context(state)
    decision: dict | None = None
    if not state.use_deterministic:
        try:
            decision = _call_llm(_SYSTEM_PROMPT.format(context=ctx), TOOL_SCHEMAS)
        except Exception as e:
            logger.warning("LLM 决策失败，降级规则引擎: %s", e)
    if decision is None:
        decision = _deterministic_agent(state)
    tn = decision.get("tool_name", "FINISHED")
    # 完整性护栏: LLM 不得在必需步骤未完成时提前收敛
    if tn == "FINISHED" and not _is_complete(state):
        logger.info("LLM 提前收敛被拦截，改用规则引擎继续")
        decision = _deterministic_agent(state)
        tn = decision.get("tool_name", "FINISHED")
    ta = decision.get("tool_args", {})
    step = TraceStep(thought=decision.get("thought", ""), action=f"decision -> {tn}", action_type=ActionType.THOUGHT)
    result = {"next_tool": tn, "next_tool_args": json.dumps(ta, ensure_ascii=False),
              "loop_count": state.loop_count + 1, "trace": _bounded_trace(state, step),
              "compressed_context": ctx}
    if tn == "FINISHED":
        result["finished"] = True
    return result

def tool_executor_node(state: FeedbackState) -> dict:
    tn = state.next_tool
    ta = json.loads(state.next_tool_args) if state.next_tool_args else {}
    if not tn or tn == "FINISHED":
        return {"finished": True}
    if tn == "classify" and not state.use_deterministic:
        result_str = classify_semantic(
            raw_text=ta.get("raw_text", state.raw_text),
            source=ta.get("source", "社区"),
            evidence_context=ta.get("evidence_context", ""),
        )
    else:
        result_str = execute_tool(tn, ta)
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    if isinstance(result, dict) and result.get("error"):
        logger.warning("工具 %s 执行异常: %s", tn, result["error"])
        obs = TraceStep(thought="", action=f"exec -> {tn}", observation=json.dumps(result, ensure_ascii=False)[:500], action_type=ActionType.OBSERVATION)
        return {"summary": result["error"], "trace": _bounded_trace(state, obs)}
    updates: dict = {}

    if tn == "classify":
        cat_m = {"作品建议": FeedbackCategory.SUGGESTION, "舆情吐槽": FeedbackCategory.SENTIMENT, "售后报修": FeedbackCategory.AFTER_SALES}
        sub_m = {"功能建议": FeedbackSubCategory.FEATURE_REQUEST, "界面优化": FeedbackSubCategory.UI_IMPROVEMENT, "内容推荐": FeedbackSubCategory.CONTENT_RECOMMEND, "产品质量": FeedbackSubCategory.PRODUCT_QUALITY, "服务态度": FeedbackSubCategory.SERVICE_ATTITUDE, "价格问题": FeedbackSubCategory.PRICE_CONCERN, "硬件故障": FeedbackSubCategory.HARDWARE_FAULT, "软件缺陷": FeedbackSubCategory.SOFTWARE_BUG, "物流问题": FeedbackSubCategory.LOGISTICS_ISSUE}
        updates["classification"] = ClassificationResult(category=cat_m.get(result.get("category", "作品建议"), FeedbackCategory.SUGGESTION), sub_category=sub_m.get(result.get("sub_category", "功能建议"), FeedbackSubCategory.FEATURE_REQUEST), tags=result.get("tags", []), confidence=result.get("confidence", 0.8))
    elif tn == "retrieve_evidence":
        updates["evidence_ids"] = [EvidenceItem(ticket_id=e.get("ticket_id", ""), content=e.get("content", ""), similarity=e.get("similarity", 0.0), category=e.get("category"), resolution=e.get("resolution")) for e in result.get("evidence", [])]
        updates["evidence_retrieved"] = True
    elif tn == "dedup":
        is_dup = result.get("is_duplicate", False)
        updates["is_duplicate"] = is_dup
        updates["dedup_bucket"] = DedupBucket(bucket_id=result.get("bucket_id", ""), representative=result.get("representative", state.raw_text), count=result.get("bucket_count", 1), members=result.get("members", []) if is_dup else [])
    elif tn == "route":
        updates["route_target"] = RouteTarget(product_line=result.get("product_line", ""), sub_line=result.get("sub_line", ""), assignee=result.get("assignee", ""), sla_hours=result.get("sla_hours", 48))
    elif tn == "push_service_guide":
        gd = result.get("guide", {})
        if gd:
            updates["service_guide"] = ServiceGuide(phone=gd.get("phone", ""), link=gd.get("link", ""), message=gd.get("message", ""))
    elif tn == "crawl_feedback":
        items = result.get("items", [])
        src_m = {"社区": FeedbackSource.COMMUNITY, "商城": FeedbackSource.MALL, "服务频道": FeedbackSource.SERVICE}
        crawled = []
        for item in items:
            lv = item.get("product_line", "其他")
            le = next((l for l in XiaomiProductLine if l.value == lv), XiaomiProductLine.OTHER)
            crawled.append(CrawledFeedback(raw_text=item.get("raw_text", ""), source=src_m.get(item.get("source", "社区"), FeedbackSource.COMMUNITY), product_line=le, url=item.get("url", ""), metadata={"batch": state.feedback_id}))
        updates["crawled_feedbacks"] = crawled
        updates["crawl_completed"] = True
        if crawled:
            updates["raw_text"] = crawled[0].raw_text
            updates["source"] = crawled[0].source
            updates["product_line"] = crawled[0].product_line
    obs = TraceStep(thought="", action=f"exec -> {tn}", observation=json.dumps(result, ensure_ascii=False)[:500], action_type=ActionType.OBSERVATION)
    updates["trace"] = _bounded_trace(state, obs)
    return updates

def reflection_node(state: FeedbackState) -> dict:
    tn = state.next_tool
    if tn == "FINISHED":
        ref = TraceStep(thought="收敛: 全部完成", action="END", action_type=ActionType.REFLECTION)
        return {"finished": True, "trace": _bounded_trace(state, ref)}
    max_loops = state.max_retries * 2 + 5
    if state.loop_count >= max_loops:
        ref = TraceStep(thought=f"收敛: 超上限({max_loops})", action="END", action_type=ActionType.REFLECTION)
        return {"finished": True, "trace": _bounded_trace(state, ref)}
    if (state.classification and state.classification.confidence < state.confidence_threshold
            and state.evidence_retrieved and state.retry_count < state.max_retries):
        ref = TraceStep(thought=f"置信度{state.classification.confidence}<{state.confidence_threshold}，第{state.retry_count+1}轮重分类", action="re-classify", action_type=ActionType.REFLECTION)
        return {"retry_count": state.retry_count + 1, "finished": False, "trace": _bounded_trace(state, ref)}
    ref = TraceStep(thought=f"步骤{tn}完成", action="continue", action_type=ActionType.REFLECTION)
    return {"trace": _bounded_trace(state, ref)}

def _should_continue(state: FeedbackState) -> Literal["tool_executor", "END"]:
    if state.next_tool == "FINISHED" or state.finished or state.loop_count > state.max_retries * 2 + 5:
        return "END"
    return "tool_executor"

def _should_loop(state: FeedbackState) -> Literal["agent_decision", "END"]:
    if state.finished or state.next_tool == "FINISHED" or state.loop_count > state.max_retries * 2 + 5:
        return "END"
    return "agent_decision"

def build_graph() -> StateGraph:
    g = StateGraph(FeedbackState)
    g.add_node("agent_decision", agent_decision_node)
    g.add_node("tool_executor", tool_executor_node)
    g.add_node("agent_reflection", reflection_node)
    g.set_entry_point("agent_decision")
    g.add_conditional_edges("agent_decision", _should_continue, {"tool_executor": "tool_executor", "END": END})
    g.add_edge("tool_executor", "agent_reflection")
    g.add_conditional_edges("agent_reflection", _should_loop, {"agent_decision": "agent_decision", "END": END})
    return g

def create_agent(checkpointer=None):
    return build_graph().compile(checkpointer=checkpointer or MemorySaver())

_SRC_MAP = {"社区": FeedbackSource.COMMUNITY, "商城": FeedbackSource.MALL, "服务频道": FeedbackSource.SERVICE}


def _invoke_state(state: FeedbackState) -> FeedbackState:
    agent = create_agent()
    result = agent.invoke(state, {"configurable": {"thread_id": f"voc-{state.feedback_id}"}})
    return FeedbackState(**result) if isinstance(result, dict) else result


def run_feedback_invoke(raw_text: str, source: str = "社区", use_llm: bool = False,
                        product_line: XiaomiProductLine = XiaomiProductLine.OTHER) -> FeedbackState:
    cfg = get_settings()
    initial = FeedbackState(
        raw_text=raw_text,
        source=_SRC_MAP.get(source, FeedbackSource.COMMUNITY),
        product_line=product_line,
        use_deterministic=not use_llm,
        max_retries=cfg.max_retries,
        confidence_threshold=cfg.confidence_threshold,
        dedup_threshold=cfg.dedup_threshold,
    )
    return _invoke_state(initial)


def run_feedback_batch(url: str | None = None, product_line: str = "",
                       source: str = "", max_items: int | None = None,
                       use_llm: bool = False) -> list[FeedbackState]:
    """爬取反馈并逐条走完整智能分发流水线，返回每条的状态。"""
    from .tools import crawl_feedback as _crawl
    cfg = get_settings()
    raw = _crawl(url or cfg.crawl_default_url, product_line=product_line, source=source,
                 max_items=max_items or cfg.crawl_max_items)
    data = json.loads(raw)
    states: list[FeedbackState] = []
    for item in data.get("items", []):
        pl = next((l for l in XiaomiProductLine if l.value == item.get("product_line")), XiaomiProductLine.OTHER)
        states.append(run_feedback_invoke(
            raw_text=item.get("raw_text", ""), source=item.get("source", "社区"),
            use_llm=use_llm, product_line=pl,
        ))
    return states