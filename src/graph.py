"""
LangGraph 状态机 —— LLM 作为唯一决策中枢

图结构:
  START -> agent_decision (LLM) -> tool_executor -> agent_reflection
              ^                      |                  |
              +--- 再决策(循环) -----+                  |
                                      END <------------+

LLM 通过 OpenAI Function Calling 绑定 6 个工具，每轮自主决定调用哪个、何时收敛。
低置信自纠错: confidence<0.7 -> LLM自主调用 retrieve_evidence -> re-classify (最多1轮)
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Literal

from .config import get_settings

logger = logging.getLogger(__name__)

# LangGraph 核心：StateGraph 构建状态机，END 标记终止，MemorySaver 做内存持久化
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 领域模型：反馈状态、追踪步骤、分类/去重/路由/售后等数据结构
from .state import FeedbackState, TraceStep, ActionType, ClassificationResult
from .state import EvidenceItem, DedupBucket, RouteTarget, ServiceGuide
from .state import FeedbackCategory, FeedbackSubCategory, FeedbackSource
from .state import XiaomiProductLine, CrawledFeedback
# 工具层：工具 Schema 定义、工具执行器、语义分类
from .tools import TOOL_SCHEMAS, execute_tool, classify_auto, small_model_predict
from .classifier.router import double_threshold_pass
from .preprocess import analyze_emoji
# 上下文压缩：将长历史摘要为 LLM 可接受的 prompt 长度
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

## 低置信自纠错流程 (最多1轮)
1. classify 返回 confidence < 0.7
2. 先判断: 是否有相关历史工单可检索?
3. 若可行 -> 调用 retrieve_evidence(query=反馈原文)
4. 拿到证据后 -> 重新 classify(raw_text=原文, evidence_context=证据摘要)
5. 若置信度仍<0.7且尚未带证据重分类 -> 进行1轮重分类; 已满1轮 -> 以最终结果收敛

## 收敛条件
- 分类非售后 + 去重完成 + 路由完成 -> FINISHED
- 分类为售后 + 去重 + 路由 + 售后电话/链接推送完成 -> FINISHED
- 置信度低但已满1轮重试 -> FINISHED

## 当前状态上下文
{context}

请用 Function Calling 选择下一步工具，或回复 {{"tool_name":"FINISHED"}} 收敛。"""

def _call_llm(system_prompt: str, tools: list[dict]) -> dict:
    """调用 LLM 进行单轮决策，决定下一步该调用哪个工具或终止流程。

    该函数是 Agent 决策循环的核心：将当前系统提示词和可用工具列表发送给 LLM，
    LLM 通过 Function Calling 自主选择下一步操作。返回值统一为一个 dict，
    包含决策描述(thought)、工具名(tool_name)、工具参数(tool_args)。

    返回值分三种情况：
    1. LLM 返回了 Function Call → 解析为 tool_name + tool_args
    2. LLM 返回纯文本，且包含 "FINISHED" → 终止信号
    3. LLM 返回纯文本且无法解析 → 兜底调用 classify 工具

    Args:
        system_prompt: 包含上下文、已执行步骤、可用工具说明的系统提示词
        tools: OpenAI Function Calling 格式的工具定义列表

    Returns:
        dict: {"thought": 决策描述, "tool_name": 工具名或"FINISHED", "tool_args": 工具参数字典}
    """
    from openai import OpenAI
    cfg = get_settings()
    client = OpenAI(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        timeout=cfg.llm_timeout,
    )
    model = cfg.model_name
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请根据当前状态，决定下一步调用哪个工具（仅一个），或回复 FINISHED。"},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=cfg.llm_temperature,  # 低温度保证决策稳定、可复现
            max_tokens=cfg.llm_max_tokens,    # 限制输出长度，仅需决策信息
        )
        msg = resp.choices[0].message
        # 记录 token 消耗，供离线评估统计单条成本
        usage = {
            "prompt": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "completion": getattr(resp.usage, "completion_tokens", 0) or 0,
        } if resp.usage else {"prompt": 0, "completion": 0}

        # 情况1：LLM 选择了某个工具调用
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}         # 参数解析失败时使用空字典兜底
            return {"thought": f"LLM决策: {tc.function.name}", "tool_name": tc.function.name, "tool_args": args, "_usage": usage}

        # 情况2：LLM 未触发 Function Call，返回纯文本内容
        content = (msg.content or "").strip()
        # 清洗可能包裹的 markdown 代码块标记
        for wrap in ("```json", "```"):
            content = content.replace(wrap, "")
        content = content.strip()

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                parsed["_usage"] = usage
            return parsed
        except json.JSONDecodeError:
            # 内容中包含 FINISHED 关键字 → 流程终止信号
            if "FINISHED" in content.upper():
                return {"thought": content, "tool_name": "FINISHED", "tool_args": {}, "_usage": usage}
            # 兜底：无法解析时默认调用 classify 进行语义分类
            return {"thought": content or "兜底分类", "tool_name": "classify", "tool_args": {}, "_usage": usage}
    except Exception as e:
        raise RuntimeError(f"LLM 调用失败: {e}") from e

def _deterministic_agent(state: FeedbackState) -> dict:
    """规则引擎降级：当 LLM 不可用时，按固定优先级级联决策下一步操作。

    作为 LLM Agent 的安全兜底，严格按流水线顺序判断：
    爬取 → 分类 → 低置信检索证据 → 重分类 → 去重 → 路由 → 售后推送 → 完成。
    低置信自纠错流程同样生效（最多 max_retries 轮），与 LLM 模式行为一致。

    Args:
        state: 当前反馈处理状态

    Returns:
        dict: 与 _call_llm 同结构的决策结果 {thought, tool_name, tool_args}
    """
    raw = state.raw_text
    src = state.source.value if hasattr(state.source, 'value') else str(state.source)
    cls = state.classification
    retry = state.retry_count

    # 步骤1：未爬取 → 先爬取反馈
    if not state.crawl_completed and not raw:
        return {"thought": "降级: 未爬取", "tool_name": "crawl_feedback",
                "tool_args": {"url": "https://www.xiaomi.cn/community/feedbacks", "max_items": 10}}
    # 步骤2：无分类结果且分类未失败过 → 执行分类
    if cls is None and "classify" not in state.failed_tools:
        return {"thought": "降级: 分类", "tool_name": "classify",
                "tool_args": {"raw_text": raw, "source": src}}
    # 分类已失败 → 无法继续路由，显式收敛并保留 errors
    if cls is None:
        return {"thought": "降级: 分类失败，终止", "tool_name": "FINISHED", "tool_args": {}}
    # 步骤3：置信度不足 + 尚未检索 + 有重试配额 → 检索历史证据
    if (cls.confidence < state.confidence_threshold and not state.evidence_retrieved
            and retry < state.max_retries and "retrieve_evidence" not in state.failed_tools):
        return {"thought": f"降级: 置信度{cls.confidence}<{state.confidence_threshold}，检索证据({retry+1}/{state.max_retries})",
                "tool_name": "retrieve_evidence", "tool_args": {"query": raw, "top_k": get_settings().evidence_top_k}}
    # 步骤4：置信度不足 + 已有证据 + 有重试配额 + 重分类仍有信息增量 → 带证据重分类
    if (cls.confidence < state.confidence_threshold and state.evidence_retrieved
            and retry < state.max_retries and not state.reclassify_stalled):
        ev = json.dumps([{"id": e.ticket_id, "content": e.content[:60], "category": e.category}
                         for e in state.evidence_ids], ensure_ascii=False)
        return {"thought": f"降级: 第{retry+1}次重分类", "tool_name": "classify",
                "tool_args": {"raw_text": raw, "source": src, "evidence_context": ev}}
    # 步骤5：尚未去重且未被标记失败 → 执行 n-gram Jaccard 去重
    if state.dedup_bucket is None and "dedup" not in state.failed_tools:
        return {"thought": "降级: 去重", "tool_name": "dedup",
                "tool_args": {"text": raw, "threshold": state.dedup_threshold}}
    # 步骤6：尚未路由且未被标记失败 → 按分类结果匹配处理人
    if state.route_target is None and "route" not in state.failed_tools:
        return {"thought": "降级: 路由", "tool_name": "route",
                "tool_args": {"category": cls.category.value, "sub_category": cls.sub_category.value,
                              "source": src, "product_line": state.product_line.value}}
    # 步骤7：售后报修类 → 推送售后电话/链接
    if (cls.category == FeedbackCategory.AFTER_SALES and state.service_guide is None
            and "push_service_guide" not in state.failed_tools):
        return {"thought": "降级: 推送售后电话/链接", "tool_name": "push_service_guide",
                "tool_args": {"feedback_text": raw, "sub_category": cls.sub_category.value, "source": src}}
    # 全部完成 → 收敛
    return {"thought": "降级: 完成", "tool_name": "FINISHED", "tool_args": {}}

def _is_complete(state: FeedbackState) -> bool:
    """完整性护栏：检查流水线必需步骤是否全部完成。

    必需步骤：
    - 分类(classification)、去重(dedup_bucket)、路由(route_target) 三者缺一不可
    - 若是售后报修类，还必须推送售后指南(service_guide)

    用于防止 LLM 在必需步骤未完成时提前返回 FINISHED。
    """
    if state.classification is None or state.dedup_bucket is None or state.route_target is None:
        return False
    if state.classification.category == FeedbackCategory.AFTER_SALES and state.service_guide is None:
        return False
    return True


def _loop_hard_limit(state: FeedbackState) -> int:
    """循环硬上限：预算耗尽后留 8 步给确定性编排收尾（去重/路由/售后推送）。"""
    return state.max_retries * 2 + 8


def _bounded_trace(state: FeedbackState, step: TraceStep) -> list[TraceStep]:
    """护栏：trace 轨迹有界，截断旧记录防止长循环/批处理导致内存无限增长。

    保留最近 max_trace_steps 条 trace 记录，丢弃更早的历史。
    """
    cfg = get_settings()
    return (list(state.trace) + [step])[-cfg.max_trace_steps:]


# ── 工具结果 → 状态字段映射：tool_executor 与 fast path 共用，字段变更只改一处 ──

def _apply_dedup(result: dict, state: FeedbackState) -> dict:
    """去重结果映射：is_duplicate + dedup_bucket。"""
    is_dup = bool(result.get("is_duplicate", False))
    return {
        "is_duplicate": is_dup,
        "dedup_bucket": DedupBucket(
            bucket_id=result.get("bucket_id", ""),
            representative=result.get("representative", state.raw_text),
            count=result.get("bucket_count", 1),
            members=result.get("members", []) if is_dup else [],
        ),
    }


def _apply_route(result: dict) -> RouteTarget:
    """路由结果映射：责任组 + SLA。"""
    return RouteTarget(
        product_line=result.get("product_line", ""), sub_line=result.get("sub_line", ""),
        assignee=result.get("assignee", ""), sla_hours=result.get("sla_hours", 48),
    )


def _apply_guide(result: dict) -> ServiceGuide | None:
    """售后指南结果映射；工具未返回 guide 时返回 None 表示不更新状态。"""
    gd = result.get("guide", {})
    if not gd:
        return None
    return ServiceGuide(phone=gd.get("phone", ""), link=gd.get("link", ""),
                        message=gd.get("message", ""))


def agent_decision_node(state: FeedbackState) -> dict:
    """Agent 决策节点：LangGraph 的入口，由 LLM 决定下一工具或终止。

    流程：
    0. 首步直调：无分类结果时下一步必然是「分类」，不消耗 LLM 调用
    1. 构建压缩上下文（将 trace 历史 + 状态摘要压缩为 prompt）
    2. 若未禁用 LLM，调用 _call_llm 获取决策；失败则降级到规则引擎
    3. 完整性护栏：若 LLM 返回 FINISHED 但必需步骤未完成，强制用规则引擎接管
    4. 记录决策步骤到 trace，更新 loop_count 和 next_tool

    Args:
        state: 当前反馈处理状态

    Returns:
        dict: 包含 next_tool、next_tool_args、loop_count、trace 等状态更新
    """
    ctx = build_compressed_context(state)
    decision: dict | None = None
    # 首步确定性直调：首次决策时（无分类结果、分类未失败过）下一步必然且仅能是
    # 「分类」，与确定性编排 step1-2 一致——这类可确定决策不消耗 LLM 调用；
    # 后续的取证/重分类/收敛判断仍交由 LLM 决策，保留 Agent 循环语义。
    if state.loop_count == 0 and state.classification is None and "classify" not in state.failed_tools:
        if state.raw_text:
            src = state.source.value if hasattr(state.source, 'value') else str(state.source)
            decision = {"thought": "首步直调: 分类(免LLM决策)",
                        "tool_name": "classify",
                        "tool_args": {"raw_text": state.raw_text, "source": src}}
        else:
            decision = _deterministic_agent(state)
    # 决策预算护栏：LLM 反复在分类/取证之间打转时，剩余步数交给确定性编排收尾，
    # 否则到达循环上限会被直接截断，去重/路由永远补不上，链路判为失败。
    # 预算 = 硬上限留 5 步收尾余量（收尾需 去重/路由/售后/完成 至多 4 步 + 缓冲）
    budget_left = _loop_hard_limit(state) - 5
    if decision is None and state.loop_count >= budget_left:
        logger.info("Agent 决策预算用尽(loop=%d)，切换到确定性编排收尾", state.loop_count)
        decision = _deterministic_agent(state)
    # 若启用 LLM 模式且仍有预算，优先使用 LLM 决策
    elif decision is None and not state.use_deterministic:
        try:
            decision = _call_llm(_SYSTEM_PROMPT.format(context=ctx), TOOL_SCHEMAS)
        except Exception as e:
            logger.warning("LLM 决策失败，降级规则引擎: %s", e)
    # LLM 失败或禁用 → 规则引擎降级
    if decision is None:
        decision = _deterministic_agent(state)
    tn = decision.get("tool_name", "FINISHED")
    # 完整性护栏：LLM 不得在必需步骤未完成时提前收敛
    if tn == "FINISHED" and not _is_complete(state):
        logger.info("LLM 提前收敛被拦截，改用规则引擎继续")
        decision = _deterministic_agent(state)
        tn = decision.get("tool_name", "FINISHED")
    ta = decision.get("tool_args", {})
    step = TraceStep(thought=decision.get("thought", ""), action=f"decision -> {tn}", action_type=ActionType.THOUGHT)
    result = {"next_tool": tn, "next_tool_args": json.dumps(ta, ensure_ascii=False),
              "loop_count": state.loop_count + 1, "trace": _bounded_trace(state, step)}
    if tn == "FINISHED":
        result["finished"] = True
    # 累计 LLM 决策的 token 消耗
    usage = decision.get("_usage")
    if usage:
        result["llm_prompt_tokens"] = state.llm_prompt_tokens + usage.get("prompt", 0)
        result["llm_completion_tokens"] = state.llm_completion_tokens + usage.get("completion", 0)
    return result

def tool_executor_node(state: FeedbackState) -> dict:
    """工具执行节点：根据 LLM/规则引擎的决策，执行对应的工具调用并更新状态。

    支持 6 种工具：
    - classify: 语义分类（LLM/级联模式下走 classify_auto）
    - retrieve_evidence: 检索历史相似工单
    - dedup: n-gram Jaccard 去重
    - route: 产品线 × 分类倒排路由
    - push_service_guide: 推送售后电话/链接
    - crawl_feedback: 爬取小米平台反馈，解析为 CrawledFeedback

    每种工具的执行结果会映射回 FeedbackState 的对应字段。

    Args:
        state: 当前反馈处理状态，包含 next_tool 和 next_tool_args

    Returns:
        dict: 包含工具执行结果映射到状态字段的更新 dict
    """
    tn = state.next_tool
    ta = json.loads(state.next_tool_args) if state.next_tool_args else {}
    # FINISHED 或空工具名 → 无需执行
    if not tn or tn == "FINISHED":
        return {"finished": True}
    # LLM 给出的工具参数属于外部输入，缺失字段回填当前状态；
    # 否则一次参数不全就会被标记为 failed_tools 永久跳过，链路再也无法收敛。
    _cls = state.classification
    _defaults = {
        "classify": {"raw_text": state.raw_text, "source": state.source.value},
        "retrieve_evidence": {"query": state.raw_text},
        "dedup": {"text": state.raw_text, "threshold": state.dedup_threshold},
        "route": {
            "category": _cls.category.value if _cls else "",
            "sub_category": _cls.sub_category.value if _cls else "",
            "source": state.source.value,
            "product_line": state.product_line.value,
        },
        "push_service_guide": {
            "feedback_text": state.raw_text,
            "sub_category": _cls.sub_category.value if _cls else "",
            "source": state.source.value,
        },
    }
    for key, val in _defaults.get(tn, {}).items():
        if ta.get(key) in (None, ""):
            ta[key] = val
    # 尚未分类就路由/推送售后 → 直接跳过（不记失败，等分类完成后再走），避免落到兜底责任组
    if tn in ("route", "push_service_guide") and _cls is None:
        obs = TraceStep(thought="", action=f"exec -> {tn}", observation="跳过: 尚未分类",
                        action_type=ActionType.OBSERVATION)
        return {"trace": _bounded_trace(state, obs)}
    # classify 在 LLM/级联模式下走智能通道，规则引擎模式下走通用执行器
    if tn == "classify" and not state.use_deterministic:
        result_str = classify_auto(
            raw_text=ta.get("raw_text", state.raw_text),
            source=ta.get("source", "社区"),
            evidence_context=ta.get("evidence_context", ""),
        )
    else:
        result_str = execute_tool(tn, ta)
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    # 工具执行异常 → 记录结构化错误并标记该步骤，避免规则引擎反复重选同一失败工具
    if isinstance(result, dict) and result.get("error"):
        msg = str(result["error"])
        logger.warning("工具 %s 执行异常: %s", tn, msg)
        obs = TraceStep(thought="", action=f"exec -> {tn}", observation=json.dumps(result, ensure_ascii=False)[:500], action_type=ActionType.OBSERVATION)
        return {
            "summary": msg,
            "errors": state.errors + [f"{tn}: {msg}"],
            "failed_tools": state.failed_tools + ([tn] if tn not in state.failed_tools else []),
            "trace": _bounded_trace(state, obs),
        }
    updates: dict = {}

    # ---- classify：三级分类(类别 + 子类别 + 标签 + 置信度) ----
    if tn == "classify":
        raw_arg = ta.get("raw_text", state.raw_text)
        src_arg = ta.get("source", "社区")
        try:
            cat = FeedbackCategory(result.get("category"))
            sub = FeedbackSubCategory(result.get("sub_category"))
        except ValueError:
            # 非法分类不静默兜底，降级规则引擎重新判定
            logger.warning("分类结果非法(%s/%s)，降级规则引擎重判",
                           result.get("category"), result.get("sub_category"))
            result = json.loads(execute_tool("classify", {"raw_text": raw_arg, "source": src_arg}))
            cat = FeedbackCategory(result["category"])
            sub = FeedbackSubCategory(result["sub_category"])
        try:
            conf = min(1.0, max(0.0, float(result.get("confidence"))))
        except (TypeError, ValueError):
            conf = 0.5  # 缺失/非法置信度按保守值处理，避免被误判为高置信
        tags = result.get("tags") if isinstance(result.get("tags"), list) else []
        updates["classification"] = ClassificationResult(
            category=cat, sub_category=sub, tags=tags, confidence=conf,
        )
        # 只有带证据的重分类才计入重试次数；结果无改进时标记停滞，避免空转
        if ta.get("evidence_context"):
            prev = state.classification
            updates["retry_count"] = state.retry_count + 1
            updates["reclassify_stalled"] = bool(
                prev is not None
                and prev.category == cat and prev.sub_category == sub
                and conf <= prev.confidence + 1e-6
            )
    # ---- retrieve_evidence：检索历史相似工单作为证据 ----
    elif tn == "retrieve_evidence":
        updates["evidence_ids"] = [EvidenceItem(ticket_id=e.get("ticket_id", ""), content=e.get("content", ""), similarity=e.get("similarity", 0.0), category=e.get("category"), resolution=e.get("resolution")) for e in result.get("evidence", [])]
        updates["evidence_retrieved"] = True
    # ---- dedup：n-gram Jaccard 去重 ----
    elif tn == "dedup":
        updates.update(_apply_dedup(result, state))
    # ---- route：按产品线 × 分类匹配处理人和 SLA ----
    elif tn == "route":
        updates["route_target"] = _apply_route(result)
    # ---- push_service_guide：推送售后电话/自助链接 ----
    elif tn == "push_service_guide":
        guide = _apply_guide(result)
        if guide is not None:
            updates["service_guide"] = guide
    # ---- crawl_feedback：爬取反馈，解析产品线和来源 ----
    elif tn == "crawl_feedback":
        items = result.get("items", [])
        crawled = []
        for item in items:
            lv = item.get("product_line", "其他")
            le = next((l for l in XiaomiProductLine if l.value == lv), XiaomiProductLine.OTHER)
            crawled.append(CrawledFeedback(raw_text=item.get("raw_text", ""), source=_SRC_MAP.get(item.get("source", "社区"), FeedbackSource.COMMUNITY), product_line=le, url=item.get("url", ""), metadata={"batch": state.feedback_id}))
        updates["crawled_feedbacks"] = crawled
        updates["crawl_completed"] = True
        if crawled:
            updates["raw_text"] = crawled[0].raw_text
            updates["source"] = crawled[0].source
            updates["product_line"] = crawled[0].product_line
    # 将执行结果追加到 trace 观测记录
    obs = TraceStep(thought="", action=f"exec -> {tn}", observation=json.dumps(result, ensure_ascii=False)[:500], action_type=ActionType.OBSERVATION)
    updates["trace"] = _bounded_trace(state, obs)
    return updates

def reflection_node(state: FeedbackState) -> dict:
    """反思/收敛判断节点：每轮工具执行完毕后，决定是继续循环、重分类还是终止。

    这是 LangGraph 状态机的核心路由节点，按优先级依次判断：
    1. LLM 已发出 FINISHED 信号 → 正常终止
    2. 循环次数超过硬上限(_loop_hard_limit) → 强制收敛，防止死循环
    3. 分类置信度不足但有证据支撑且未达重试上限 → 触发低置信自纠错，重新分类
    4. 以上均不满足 → 继续下一轮 Agent 决策循环

    Args:
        state: 当前反馈处理的完整状态

    Returns:
        dict: 包含 finished(是否终止)、trace(追踪记录)、可选 retry_count 的状态更新
    """
    tn = state.next_tool

    # 优先级1：LLM 主动返回 FINISHED，流程正常收敛
    if tn == "FINISHED":
        ref = TraceStep(thought="收敛: 全部完成", action="END", action_type=ActionType.REFLECTION)
        return {"finished": True, "trace": _bounded_trace(state, ref)}

    # 优先级2：循环次数超硬上限，强制终止避免无限循环（预算耗尽后由确定性编排收尾，留足收尾步数）
    max_loops = _loop_hard_limit(state)
    if state.loop_count >= max_loops:
        ref = TraceStep(thought=f"收敛: 超上限({max_loops})", action="END", action_type=ActionType.REFLECTION)
        return {"finished": True, "trace": _bounded_trace(state, ref)}

    # 优先级3：低置信自纠错 —— 置信度不足 + 已有检索证据 + 仍有重试配额 → 下一轮重分类
    # 重试计数由实际执行带证据重分类的节点递增，避免未真正重分类却消耗配额
    if (state.classification and state.classification.confidence < state.confidence_threshold
            and state.evidence_retrieved and state.retry_count < state.max_retries
            and not state.reclassify_stalled):
        ref = TraceStep(
            thought=f"置信度{state.classification.confidence}<{state.confidence_threshold}，第{state.retry_count+1}轮重分类",
            action="re-classify",
            action_type=ActionType.REFLECTION,
        )
        return {"finished": False, "trace": _bounded_trace(state, ref)}

    # 优先级4：继续循环 —— 当前步骤正常完成，进入下一轮 LLM 决策
    ref = TraceStep(thought=f"步骤{tn}完成", action="continue", action_type=ActionType.REFLECTION)
    return {"trace": _bounded_trace(state, ref)}

def _should_continue(state: FeedbackState) -> Literal["tool_executor", "END"]:
    """条件边路由：Agent 决策后，判断是进入工具执行还是终止。

    当 LLM 返回 FINISHED、状态已标记 finished、或循环超限时 → END，否则 → tool_executor。
    """
    if state.next_tool == "FINISHED" or state.finished or state.loop_count >= _loop_hard_limit(state):
        return "END"
    return "tool_executor"

def _should_loop(state: FeedbackState) -> Literal["agent_decision", "END"]:
    """条件边路由：反思后，判断是回到 Agent 再决策还是终止。

    当 finished 或 FINISHED 或循环超限时 → END，否则 → agent_decision 继续循环。
    """
    if state.finished or state.next_tool == "FINISHED" or state.loop_count >= _loop_hard_limit(state):
        return "END"
    return "agent_decision"

def build_graph() -> StateGraph:
    """构建 LangGraph 状态机，定义节点和条件边。

    图结构：
        START → agent_decision ──→ tool_executor → agent_reflection
                  ↑    ↓ (FINISHED/超限)              ↑        ↓
                  └──── 循环 ─────────────────────────┘      END

    三个节点：
    - agent_decision: LLM/规则引擎决策
    - tool_executor: 执行选中的工具
    - agent_reflection: 反思收敛判断
    """
    g = StateGraph(FeedbackState)
    g.add_node("agent_decision", agent_decision_node)
    g.add_node("tool_executor", tool_executor_node)
    g.add_node("agent_reflection", reflection_node)
    g.set_entry_point("agent_decision")
    # Agent 决策后 → 工具执行 或 终止
    g.add_conditional_edges("agent_decision", _should_continue, {"tool_executor": "tool_executor", "END": END})
    # 工具执行后 → 反思
    g.add_edge("tool_executor", "agent_reflection")
    # 反思后 → 再决策(循环) 或 终止
    g.add_conditional_edges("agent_reflection", _should_loop, {"agent_decision": "agent_decision", "END": END})
    return g

def create_agent(checkpointer=None):
    """创建编译后的 Agent，支持可选的外部 checkpointer（默认 MemorySaver）。

    每次调用 invoke 时，以 feedback_id 作为 thread_id 隔离不同反馈的会话上下文。
    """
    return build_graph().compile(checkpointer=checkpointer or MemorySaver())

# 反馈来源中文名 → FeedbackSource 枚举映射
_SRC_MAP = {"社区": FeedbackSource.COMMUNITY, "商城": FeedbackSource.MALL, "服务频道": FeedbackSource.SERVICE}


def _invoke_state(state: FeedbackState) -> FeedbackState:
    """内部调用：创建 Agent 实例并以 feedback_id 为 thread_id 执行一次完整流水线。"""
    agent = create_agent()
    result = agent.invoke(state, {"configurable": {"thread_id": f"voc-{state.feedback_id}"}})
    return FeedbackState(**result) if isinstance(result, dict) else result


def _try_fast_path(raw_text: str, source: str, product_line: XiaomiProductLine) -> FeedbackState | None:
    """小模型双阈值达标时整条链路直通，完全不进入 LLM 决策循环。

    分类直接采用小模型结果，随后确定性执行 dedup → route → (售后推送)。
    小模型缺失、置信度不达标或任一步工具异常时返回 None，交回原有 Agent 流程。
    """
    cfg = get_settings()
    if not cfg.enable_fast_path:
        return None
    profile = analyze_emoji(raw_text)
    # 纯表情没有可判定的文本语义，交回 Agent/LLM 流程
    if not profile.text.strip():
        return None
    text = profile.text
    pred = small_model_predict(text)
    # 双阈值判定与 CascadeRouter 共用同一实现，避免两处口径漂移
    if not double_threshold_pass(pred, cfg.small_model_category_threshold,
                                 cfg.small_model_sub_threshold):
        return None

    cat = next((c for c in FeedbackCategory if c.value == pred.category), None)
    sub = next((s for s in FeedbackSubCategory if s.value == pred.sub_category), None)
    if cat is None or sub is None:
        logger.warning("fast path 跳过: 小模型输出不在分类体系内 %s/%s", pred.category, pred.sub_category)
        return None

    state = FeedbackState(
        raw_text=text,
        original_text=raw_text,
        emoji_count=profile.count,
        emoji_strong_negative=profile.strong_negative,
        emoji_positive=profile.positive,
        emoji_ambiguous=profile.ambiguous,
        emoji_only=profile.only_emoji,
        source=_SRC_MAP.get(source, FeedbackSource.COMMUNITY),
        product_line=product_line,
        use_deterministic=True,
        fast_path=True,
        confidence_threshold=cfg.confidence_threshold,
        dedup_threshold=cfg.dedup_threshold,
    )
    state.classification = ClassificationResult(
        category=cat, sub_category=sub, tags=["小模型", "直通"],
        confidence=round(min(pred.category_confidence, pred.sub_confidence), 4),
    )
    state.trace.append(TraceStep(
        thought=f"fast_path: 小模型直出 {cat.value}/{sub.value} "
                f"({pred.category_confidence:.2f}/{pred.sub_confidence:.2f})",
        action="fast_path", action_type=ActionType.THOUGHT,
    ))

    def _run(tool: str, args: dict) -> dict | None:
        payload = json.loads(execute_tool(tool, args))
        if isinstance(payload, dict) and payload.get("error"):
            logger.warning("fast path 工具 %s 失败，回退 Agent 流程: %s", tool, payload["error"])
            return None
        state.trace.append(TraceStep(
            thought="", action=f"exec -> {tool}",
            observation=json.dumps(payload, ensure_ascii=False)[:500],
            action_type=ActionType.OBSERVATION,
        ))
        return payload

    d = _run("dedup", {"text": text, "threshold": cfg.dedup_threshold})
    if d is None:
        return None
    dedup_updates = _apply_dedup(d, state)
    state.is_duplicate = dedup_updates["is_duplicate"]
    state.dedup_bucket = dedup_updates["dedup_bucket"]

    r = _run("route", {"category": cat.value, "sub_category": sub.value,
                       "source": source, "product_line": product_line.value})
    if r is None:
        return None
    state.route_target = _apply_route(r)

    if cat == FeedbackCategory.AFTER_SALES:
        g = _run("push_service_guide", {
            "feedback_text": text, "sub_category": sub.value, "source": source,
        })
        if g is None:
            return None
        guide = _apply_guide(g)
        if guide is not None:
            state.service_guide = guide

    state.finished = True
    state.next_tool = "FINISHED"
    state.trace.append(TraceStep(
        thought="收敛: fast path 完成", action="END", action_type=ActionType.REFLECTION,
    ))
    state.trace = state.trace[-cfg.max_trace_steps:]
    logger.info("fast path 完成[%s]: %s/%s (零 LLM 调用)", raw_text[:20], cat.value, sub.value)
    return state


def run_feedback_invoke(raw_text: str, source: str = "社区", use_llm: bool = False,
                        product_line: XiaomiProductLine = XiaomiProductLine.OTHER,
                        feedback_id: str = "") -> FeedbackState:
    """单条反馈入口：传入原始文本，走完完整智能分发流水线，返回最终状态。

    启用 LLM 模式时会先尝试 fast path：小模型双阈值达标则整链路确定性完成，
    不发起任何 LLM 调用；否则进入 LangGraph Agent 循环（LLM 决策 + 规则兜底）。

    Args:
        raw_text: 用户反馈原始文本
        source: 反馈来源 ("社区" / "商城" / "服务频道")
        use_llm: 是否启用 LLM 决策（False 则纯规则引擎）
        product_line: 小米产品线枚举

    Returns:
        FeedbackState: 包含分类、去重、路由、售后等完整结果的状态对象
    """
    cfg = get_settings()
    if use_llm:
        fast = _try_fast_path(raw_text, source, product_line)
        if fast is not None:
            return fast
    profile = analyze_emoji(raw_text)
    # 纯表情时保留原文，交给 LLM 判断；否则链路统一使用剥离后的文本
    text = profile.text if profile.text.strip() else raw_text
    initial = FeedbackState(
        feedback_id=feedback_id or f"FB-{uuid.uuid4().hex[:8].upper()}",
        raw_text=text,
        original_text=raw_text,
        emoji_count=profile.count,
        emoji_strong_negative=profile.strong_negative,
        emoji_positive=profile.positive,
        emoji_ambiguous=profile.ambiguous,
        emoji_only=profile.only_emoji,
        source=_SRC_MAP.get(source, FeedbackSource.COMMUNITY),
        product_line=product_line,
        use_deterministic=not use_llm,    # use_llm=False → 规则引擎模式
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