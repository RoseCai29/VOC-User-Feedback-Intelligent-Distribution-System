"""
三层上下文压缩策略 —— "摘要 + 统计量 + 轨迹"
在保证决策信息量的同时控制 token 开销。
"""

from __future__ import annotations

from .state import FeedbackState, TraceStep


def compute_statistics(state: FeedbackState) -> dict:
    """从状态中提取关键统计量"""
    stats = {
        "trace_steps": len(state.trace),
        "evidence_count": len(state.evidence_ids),
        "retry_count": state.retry_count,
        "has_classification": state.classification is not None,
        "is_duplicate": state.is_duplicate,
        "has_route": state.route_target is not None,
        "has_service_guide": state.service_guide is not None,
        "finished": state.finished,
    }
    if state.classification:
        stats["classification_confidence"] = state.classification.confidence
    return stats


def generate_summary(state: FeedbackState) -> str:
    """生成压缩摘要 —— 用关键信息替代完整文本"""
    cls = state.classification
    rt = state.route_target
    guide = state.service_guide

    parts = [
        f"[反馈] {state.raw_text[:80]}{'...' if len(state.raw_text) > 80 else ''}",
        f"[来源] {state.source.value}",
    ]

    if cls:
        parts.append(f"[分类] {cls.category.value} → {cls.sub_category.value} (置信度: {cls.confidence})")
        if cls.tags:
            parts.append(f"[标签] {', '.join(cls.tags)}")

    if state.evidence_ids:
        parts.append(f"[证据] {len(state.evidence_ids)}条历史工单")

    if state.is_duplicate and state.dedup_bucket:
        parts.append(f"[去重] 归入桶 {state.dedup_bucket.bucket_id} (共{state.dedup_bucket.count}条)")

    if rt:
        parts.append(f"[路由] {rt.product_line}/{rt.sub_line} → {rt.assignee} (SLA: {rt.sla_hours}h)")

    if guide:
        parts.append(f"[售后渠道] 电话:{guide.phone} 链接:{guide.link}")

    return "\n".join(parts)


def compress_trace(trace: list[TraceStep], max_steps: int = 5) -> str:
    """压缩轨迹日志 —— 只保留最近的 N 步 + 首步"""
    if not trace:
        return "[轨迹] 空"

    lines = []
    steps_to_show = trace[:1]  # 始终保留第一步
    if len(trace) > max_steps:
        steps_to_show += trace[-(max_steps - 1):]
    else:
        steps_to_show = trace

    for step in steps_to_show:
        lines.append(
            f"  Step{step.step_id}: "
            f"[{step.action_type.value}] {step.thought[:60]}..."
        )

    if len(trace) > max_steps:
        lines.append(f"  ... 省略 {len(trace) - max_steps} 步")

    return "[轨迹]\n" + "\n".join(lines)


def build_compressed_context(state: FeedbackState, max_trace_steps: int = 5) -> str:
    """
    三层压缩的完整上下文构建。
    格式: 摘要 + 统计量 + 轨迹
    """
    stats = compute_statistics(state)
    summary = generate_summary(state)
    trace_str = compress_trace(state.trace, max_trace_steps)

    context = (
        f"## 摘要\n{summary}\n\n"
        f"## 统计量\n{stats}\n\n"
        f"## {trace_str}"
    )

    return context