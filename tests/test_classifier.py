"""小模型与级联路由测试。"""

import json

import pytest

from src.classifier.base import SmallModelPrediction
from src.classifier.linear import LinearSmallModel
from src.classifier.router import CascadeRouter
from src.evaluate import evaluate


_SEEDS = {
    ("作品建议", "功能建议"): "希望增加新功能",
    ("作品建议", "界面优化"): "界面布局不好看",
    ("作品建议", "内容推荐"): "推荐算法不准确",
    ("舆情吐槽", "产品质量"): "产品质量太差了",
    ("舆情吐槽", "服务态度"): "客服态度很差",
    ("舆情吐槽", "价格问题"): "价格太贵了",
    ("售后报修", "硬件故障"): "手机无法开机",
    ("售后报修", "软件缺陷"): "APP闪退崩溃",
    ("售后报修", "物流问题"): "物流太慢还没发货",
}


def _build_samples(n_per: int = 12):
    texts, cats, subs = [], [], []
    for (cat, sub), seed in _SEEDS.items():
        for i in range(n_per):
            texts.append(f"{seed}，第{i}条反馈")
            cats.append(cat)
            subs.append(sub)
    return texts, cats, subs


def test_linear_small_model_train_save_load(tmp_path):
    texts, cats, subs = _build_samples()
    model = LinearSmallModel.train(texts, cats, subs)
    path = tmp_path / "small.joblib"
    model.save(path)

    loaded = LinearSmallModel.load(path)
    pred = loaded.predict("APP老是闪退崩溃，根本用不了")
    assert pred is not None
    assert pred.category == "售后报修"
    assert pred.sub_category == "软件缺陷"
    assert 0.0 <= pred.category_confidence <= 1.0
    assert 0.0 <= pred.sub_confidence <= 1.0


class _FakeModel:
    def __init__(self, pred: SmallModelPrediction):
        self.pred = pred

    def is_ready(self) -> bool:
        return True

    def predict(self, text: str) -> SmallModelPrediction:
        return self.pred


def test_router_uses_small_model_when_confident():
    pred = SmallModelPrediction("售后报修", "软件缺陷", 0.95, 0.90)
    router = CascadeRouter(_FakeModel(pred), 0.85, 0.70, fallback=lambda *a: "{}")
    out = json.loads(router.classify("APP闪退"))
    assert out["engine"] == "small"
    assert out["escalated"] is False
    assert out["category"] == "售后报修"


def test_router_escalates_when_low_confidence():
    pred = SmallModelPrediction("售后报修", "软件缺陷", 0.60, 0.45)
    called = {"hit": False}

    def fallback(text, source, evidence):
        called["hit"] = True
        return json.dumps({
            "category": "售后报修", "sub_category": "软件缺陷",
            "tags": [], "confidence": 0.88, "engine": "llm",
        }, ensure_ascii=False)

    router = CascadeRouter(_FakeModel(pred), 0.85, 0.70, fallback=fallback)
    out = json.loads(router.classify("这个APP不好用"))
    assert called["hit"] is True
    assert out["engine"] == "llm"
    assert out["escalated"] is True
    assert out["confidence"] == 0.88


def test_classify_auto_falls_back_when_model_missing(monkeypatch, tmp_path):
    from src import tools
    from src.config import get_settings

    cfg = get_settings()
    monkeypatch.setattr(cfg, "classifier_mode", "cascade")
    monkeypatch.setattr(cfg, "cascade_fallback", "rules")
    monkeypatch.setattr(cfg, "small_model_path", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(tools, "_small_model", None)
    monkeypatch.setattr(tools, "_small_model_failed", False)

    out = json.loads(tools.classify_auto("客服态度太差了"))
    assert out["engine"] == "rules"
    assert out["category"] == "舆情吐槽"


def test_graph_routes_classify_through_cascade(monkeypatch):
    """非规则引擎模式下，graph 的 classify 必须走 classify_auto 级联通道。"""
    from src import graph
    from src.state import FeedbackState

    calls = {"auto": 0}

    def fake_auto(raw_text, source="社区", evidence_context=""):
        calls["auto"] += 1
        return json.dumps({
            "category": "售后报修", "sub_category": "软件缺陷",
            "tags": ["小模型"], "confidence": 0.9, "engine": "small", "escalated": False,
        }, ensure_ascii=False)

    monkeypatch.setattr(graph, "classify_auto", fake_auto)
    args = json.dumps({"raw_text": "APP闪退", "source": "社区"})

    llm_state = FeedbackState(raw_text="APP闪退", next_tool="classify",
                              next_tool_args=args, use_deterministic=False)
    updates = graph.tool_executor_node(llm_state)
    assert calls["auto"] == 1
    assert updates["classification"].sub_category.value == "软件缺陷"

    # 规则引擎模式直接走通用执行器，不触发级联
    rules_state = FeedbackState(raw_text="APP闪退", next_tool="classify",
                                next_tool_args=args, use_deterministic=True)
    graph.tool_executor_node(rules_state)
    assert calls["auto"] == 1


def test_evaluate_reports_cascade_metrics():
    examples = [
        {"text": "a", "category": "售后报修", "sub_category": "软件缺陷"},
        {"text": "b", "category": "作品建议", "sub_category": "功能建议"},
    ]

    def fake_label(text, source="社区"):
        if text == "a":
            return {"category": "售后报修", "sub_category": "软件缺陷", "engine": "small", "escalated": False}
        return {"category": "作品建议", "sub_category": "功能建议", "engine": "llm", "escalated": True}

    result = evaluate(fake_label, examples)
    assert result["coverage"] == 0.5
    assert result["escalation_rate"] == 0.5
    assert result["precision_at_coverage_l1"] == 1.0
    assert result["precision_at_coverage_l2"] == 1.0
    assert result["engine_breakdown"] == {"small": 1, "llm": 1}
