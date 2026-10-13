"""
VOC 用户反馈智能分发系统 —— 演示入口
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError: pass

from src.graph import run_feedback_invoke, run_feedback_batch
from src.state import FeedbackState, XiaomiProductLine
from src.tools import reset_dedup_buckets

DEMO_FEEDBACKS = [
    {"text": "", "source": "社区", "desc": "★ 爬虫采集: 小米社区全量爬取 -> 自动识别产品线 -> 逐条处理"},
    {"text": "APP经常闪退崩溃，打开作品页面就黑屏了，需要售后保修尽快处理！", "source": "社区", "desc": "软件缺陷 -> 路由+售后电话/链接推送"},
    {"text": "强烈建议增加夜间暗黑模式，晚上刷社区眼睛太累了", "source": "社区", "desc": "界面优化建议 -> 路由UED"},
    {"text": "产品质量太差了，买回来用了不到三天就坏了，售后推三阻四不处理！", "source": "商城", "desc": "舆情吐槽 -> 产品质量 -> 紧急处理"},
    {"text": "物流等了一周还没到，客服电话打不通，太让人失望了", "source": "服务频道", "desc": "物流问题 -> 售后电话/链接"},
    {"text": "买的设备无法开机，充电指示灯不亮，需要售后维修", "source": "商城", "desc": "硬件故障 -> 售后电话/链接推送"},
    {"text": "这个APP真的不好用啊，打开总是出毛病", "source": "社区", "desc": "★ 低置信自纠错(3次循环): 置信度0.45->取证->重分类0.88"},
    {"text": "软件最近总是用不了，不知道什么情况", "source": "服务频道", "desc": "★ 低置信自纠错: 模糊反馈->检索证据->纠正分类"},
]

def sep(title="", w=70):
    p = max(0, (w - len(title) - 2) // 2) if title else 0
    print(f"\n{'='*p} {title} {'='*p}" if title else "="*w)

def print_state(s: FeedbackState):
    c = s.classification; r = s.route_target; g = s.service_guide; d = s.dedup_bucket
    print(f"  ID:{s.feedback_id} | Src:{s.source.value} | Line:{s.product_line.value}")
    if s.crawled_feedbacks: print(f"  Crawled: {len(s.crawled_feedbacks)}条")
    if c: print(f"  Class: {c.category.value}->{c.sub_category.value} | Conf:{c.confidence} | Tags:{c.tags}")
    if s.evidence_ids: print(f"  Evidence: {len(s.evidence_ids)}条")
    if d: print(f"  Dedup: {d.bucket_id} (x{d.count}) | Dup:{s.is_duplicate}")
    if r: print(f"  Route: {r.product_line}/{r.sub_line} -> {r.assignee} | SLA:{r.sla_hours}h")
    if g: print(f"  Guide: 电话{g.phone} | Link:{g.link}")
    print(f"  Trace: {len(s.trace)}步 | Retry:{s.retry_count} | Done:{s.finished}")
    print("-"*60)

def print_trace(s: FeedbackState):
    sep("全链路审计轨迹 Trace")
    for i, st in enumerate(s.trace):
        print(f"  [{i+1}] {st.action_type.value.upper()} | {st.step_id}")
        if st.thought: print(f"    T: {st.thought}")
        if st.action: print(f"    A: {st.action}")
        if st.observation: print(f"    O: {st.observation[:200]}")

def run_demo():
    use_llm = os.getenv("USE_LLM", "").lower() == "true"
    sep(f"VOC 智能分发 ({'LLM' if use_llm else '确定性代理'})")
    print(f"  反馈: {len(DEMO_FEEDBACKS)}条 | 工具: classify/retrieve_evidence/dedup/route/push_service_guide/crawl_feedback")
    reset_dedup_buckets()

    for i, fb in enumerate(DEMO_FEEDBACKS):
        sep(f"[{i+1}/{len(DEMO_FEEDBACKS)}] {fb['desc']}")
        print(f"  Text: {fb['text'][:60] if fb['text'] else '(crawl触发)'}...")
        try:
            if not fb["text"]:
                states = run_feedback_batch(use_llm=use_llm)
                for st in states:
                    print_state(st)
                    print_trace(st)
                print(f"  >> 共 {len(states)} 条爬取反馈已逐条处理")
            else:
                state = run_feedback_invoke(raw_text=fb["text"], source=fb["source"], use_llm=use_llm, product_line=XiaomiProductLine.OTHER)
                print_state(state)
                print_trace(state)
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
    sep("Done")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "server", "info"], default="demo")
    p.add_argument("--llm", action="store_true")
    args = p.parse_args()
    if args.llm:
        os.environ["USE_LLM"] = "true"

    if args.mode == "info":
        sep("LangGraph 图结构")
        print("  decision -> execute -> reflect -> (loop) -> END")
        print("  6 tools: classify|retrieve_evidence|dedup|route|push_service_guide|crawl_feedback")
        print("  低置信自纠错: confidence<0.7 -> retrieve_evidence -> re-classify (max 3轮)")
        print("  启动 API: python -m src.main --mode server")
        print("  或: uvicorn src.api:app --reload")
    elif args.mode == "server":
        from src.logging_config import setup_logging; setup_logging()
        import uvicorn
        from src.config import get_settings
        cfg = get_settings()
        print(f"VOC API: http://{cfg.server_host}:{cfg.server_port}")
        print(f"Swagger: http://{cfg.server_host}:{cfg.server_port}/docs")
        uvicorn.run("src.api:app", host=cfg.server_host, port=cfg.server_port, reload=True)
    else:
        run_demo()