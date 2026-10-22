"""
Agent 端到端评测 —— 跑完整 LangGraph 流水线，产出 Agent 工程指标。

与 src/evaluate.py 的分工:
  evaluate.py 只评「分类器」；本脚本评「整条链路」，含降级率、步数、自纠错与 token 成本。

用法:
    # 规则引擎模式（无需 API Key）
    python -m scripts.eval_agent --dataset data/eval_set_hard.json

    # LLM 决策模式（需在 .env 配置 OPENAI_API_KEY / MODEL_NAME）
    python -m scripts.eval_agent --dataset data/eval_set_hard.json --llm

指标口径:
  pipeline_success_rate  分类/去重/路由三件套齐全且流程收敛的占比
  degraded_rate          出现工具异常并记录 errors 的占比
  avg_steps              单条反馈的 trace 平均步数
  escalation_rate        触发过低置信重分类的占比
  token 成本             仅在 --llm 下有效，来自每次 LLM 决策的 usage
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from src.config import get_settings
from src.db import init_db
from src.graph import run_feedback_invoke
from src.tools import reset_dedup_buckets


def _load_examples(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(p.read_text(encoding="utf-8"))["examples"]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return sorted(values)[min(len(values) - 1, int(0.95 * len(values)))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent 端到端评测")
    ap.add_argument("--dataset", default="data/eval_set_hard.json", help="评估集(json/jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全量")
    ap.add_argument("--llm", action="store_true", help="启用 LLM 决策(需配置 API Key)")
    ap.add_argument("--out", default="", help="指标写入的 JSON 路径(可选)")
    args = ap.parse_args()

    cfg = get_settings()
    cfg.enable_pending_pool = False  # 离线评估不写入待标注池
    cfg.enable_live_crawl = False
    init_db()
    reset_dedup_buckets()

    examples = _load_examples(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit("评估集为空")

    l1 = l2 = success = degraded = escalated = fast = 0
    steps: list[int] = []
    loops: list[int] = []
    retries: list[int] = []
    latencies: list[float] = []
    prompt_tokens = completion_tokens = 0
    errors: Counter = Counter()
    misfits: list[dict] = []

    for e in examples:
        t0 = time.time()
        state = run_feedback_invoke(
            raw_text=e["text"], source=e.get("source", "社区"), use_llm=args.llm,
        )
        latencies.append((time.time() - t0) * 1000)

        cls = state.classification
        pred_c = cls.category.value if cls else None
        pred_s = cls.sub_category.value if cls else None
        gold_c, gold_s = e["category"], e["sub_category"]
        if pred_c == gold_c:
            l1 += 1
            if pred_s == gold_s:
                l2 += 1
        else:
            misfits.append({"text": e["text"][:40], "gold": f"{gold_c}/{gold_s}",
                            "pred": f"{pred_c}/{pred_s}"})

        if state.finished and cls is not None and state.route_target is not None:
            success += 1
        if state.errors:
            degraded += 1
            for err in state.errors:
                errors[str(err).split(":")[0]] += 1
        if state.retry_count > 0:
            escalated += 1
        if state.fast_path:
            fast += 1

        steps.append(len(state.trace))
        loops.append(state.loop_count)
        retries.append(state.retry_count)
        prompt_tokens += state.llm_prompt_tokens
        completion_tokens += state.llm_completion_tokens

    n = len(examples)
    report = {
        "dataset": args.dataset,
        "mode": "llm" if args.llm else "rules",
        "total": n,
        "level1_accuracy": round(l1 / n, 4),
        "level2_accuracy": round(l2 / n, 4),
        "pipeline_success_rate": round(success / n, 4),
        "degraded_rate": round(degraded / n, 4),
        "degraded_by_tool": dict(errors),
        "escalation_rate": round(escalated / n, 4),
        "fast_path_rate": round(fast / n, 4),
        "avg_steps": round(statistics.mean(steps), 2),
        "p95_steps": _p95([float(s) for s in steps]),
        "max_retry_observed": max(retries) if retries else 0,
        "avg_loops": round(statistics.mean(loops), 2),
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "p95_latency_ms": round(_p95(latencies), 1),
        "llm_prompt_tokens_total": prompt_tokens,
        "llm_completion_tokens_total": completion_tokens,
        "avg_llm_tokens_per_case": round((prompt_tokens + completion_tokens) / n, 1),
        "misclassified": misfits,
    }

    out = json.dumps(report, ensure_ascii=False, indent=2)
    print(out)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
