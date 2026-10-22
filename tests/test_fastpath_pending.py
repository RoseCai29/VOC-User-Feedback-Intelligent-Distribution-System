"""fast path、待标注池、伪标签冷启动、真实数据集字段适配的测试。"""

import json

from src.classifier.base import SmallModelPrediction


class _FakeModel:
    def __init__(self, pred: SmallModelPrediction | None):
        self.pred = pred

    def is_ready(self) -> bool:
        return True

    def predict(self, text: str):
        return self.pred


def _high_conf():
    return SmallModelPrediction("售后报修", "软件缺陷", 0.95, 0.90)


def _low_conf():
    return SmallModelPrediction("售后报修", "软件缺陷", 0.55, 0.40)


class TestFastPath:
    def test_zero_llm_full_pipeline(self, monkeypatch):
        from src import graph
        from src.config import get_settings

        monkeypatch.setattr(get_settings(), "enable_fast_path", True)
        monkeypatch.setattr(graph, "small_model_predict", lambda text: _high_conf())

        def _must_not_run(state):
            raise AssertionError("fast path 不应进入 LangGraph Agent 循环")

        monkeypatch.setattr(graph, "_invoke_state", _must_not_run)
        state = graph.run_feedback_invoke("APP闪退崩溃，设备无法开机", "社区", use_llm=True)

        assert state.fast_path is True
        assert state.finished is True
        assert state.classification.category.value == "售后报修"
        assert state.classification.sub_category.value == "软件缺陷"
        assert state.dedup_bucket is not None
        assert state.route_target is not None
        assert state.service_guide is not None
        assert any(step.action == "fast_path" for step in state.trace)

    def test_low_confidence_falls_back_to_agent(self, monkeypatch):
        from src import graph
        from src.config import get_settings

        monkeypatch.setattr(get_settings(), "enable_fast_path", True)
        monkeypatch.setattr(graph, "small_model_predict", lambda text: _low_conf())
        called = {"hit": False}

        def _fake_invoke(state):
            called["hit"] = True
            return state

        monkeypatch.setattr(graph, "_invoke_state", _fake_invoke)
        state = graph.run_feedback_invoke("这个APP不太好用", "社区", use_llm=True)

        assert called["hit"] is True
        assert state.fast_path is False

    def test_missing_model_skips_fast_path(self, monkeypatch):
        from src import graph
        from src.config import get_settings

        monkeypatch.setattr(get_settings(), "enable_fast_path", True)
        monkeypatch.setattr(graph, "small_model_predict", lambda text: None)
        called = {"hit": False}

        def _fake_invoke(state):
            called["hit"] = True
            return state

        monkeypatch.setattr(graph, "_invoke_state", _fake_invoke)
        graph.run_feedback_invoke("冷启动期没有小模型", "社区", use_llm=True)
        assert called["hit"] is True


class TestPendingPool:
    def test_idempotent_write(self):
        from src.db import add_pending_label, count_pending_labels, update_pending_label

        before = count_pending_labels()
        for _ in range(2):
            add_pending_label(
                text_hash="test-hash-idempotent", raw_text="同一个低置信样本",
                source="社区", small_category="售后报修", small_sub_category="软件缺陷",
                small_category_confidence=0.5, small_sub_confidence=0.4,
                llm_category="售后报修", llm_sub_category="硬件故障",
                llm_confidence=0.7, llm_engine="llm",
            )
        assert count_pending_labels() == before + 1

        update_pending_label("test-hash-idempotent", "舆情吐槽", "产品质量", "approved")
        add_pending_label(
            text_hash="test-hash-idempotent", raw_text="同一个低置信样本",
            llm_category="作品建议", llm_sub_category="功能建议",
        )
        from src.db import list_pending_labels
        rec = next(r for r in list_pending_labels() if r.text_hash == "test-hash-idempotent")
        assert rec.status == "approved"
        assert rec.human_category == "舆情吐槽"
        assert rec.llm_category == "作品建议"

    def test_cascade_escalation_captured(self, monkeypatch):
        from src import tools
        from src.config import get_settings
        from src.db import count_pending_labels

        cfg = get_settings()
        monkeypatch.setattr(cfg, "classifier_mode", "cascade")
        monkeypatch.setattr(cfg, "cascade_fallback", "rules")
        monkeypatch.setattr(cfg, "enable_pending_pool", True)
        monkeypatch.setattr(tools, "_get_small_model", lambda: _FakeModel(_low_conf()))

        before = count_pending_labels()
        out = json.loads(tools.classify_auto("客服态度差到我不想再用这个产品了-唯一低置信样本"))
        assert out["escalated"] is True
        assert count_pending_labels() == before + 1

        monkeypatch.setattr(tools, "_get_small_model", lambda: _FakeModel(_high_conf()))
        out2 = json.loads(tools.classify_auto("APP闪退崩溃无法进入系统-唯一高置信样本"))
        assert out2["engine"] == "small"
        assert count_pending_labels() == before + 1


class TestBootstrap:
    def test_pseudo_label_filters(self):
        from src.classifier.bootstrap import pseudo_label

        examples = [
            {"text": "a", "source": "社区"},
            {"text": "b", "source": "社区"},
            {"text": "c", "source": "社区"},
            {"text": "a", "source": "社区"},
        ]

        def fake_classifier(text, source="社区"):
            if text == "a":
                return json.dumps({"category": "售后报修", "sub_category": "软件缺陷",
                                   "confidence": 0.92, "engine": "llm"}, ensure_ascii=False)
            if text == "b":
                return json.dumps({"category": "作品建议", "sub_category": "功能建议",
                                   "confidence": 0.60, "engine": "llm"}, ensure_ascii=False)
            return json.dumps({"category": "舆情吐槽", "sub_category": "价格问题",
                               "confidence": 0.80, "engine": "rules"}, ensure_ascii=False)

        rows, stats = pseudo_label(examples, 0.8, classifier=fake_classifier)
        assert [r["text"] for r in rows] == ["a"]
        assert rows[0]["label_type"] == "pseudo"
        assert stats["duplicate_texts"] == 1
        assert stats["dropped_low_confidence"] == 1
        assert stats["dropped_non_llm"] == 1

    def test_from_pending_export_format(self, tmp_path):
        from src.classifier.bootstrap import export_pending
        from src.db import add_pending_label

        add_pending_label(text_hash="test-hash-export", raw_text="导出样本", source="商城")
        report = export_pending(tmp_path / "pending.jsonl", "pending")
        assert report["exported"] >= 1
        lines = (tmp_path / "pending.jsonl").read_text(encoding="utf-8").splitlines()
        row = next(json.loads(l) for l in lines if json.loads(l)["text_hash"] == "test-hash-export")
        assert row["text"] == "导出样本"
        assert row["status"] == "pending"


class TestDatasetLoader:
    def test_csv_field_detection_and_alias(self, tmp_path):
        from src.classifier.dataset import load_examples

        path = tmp_path / "real.csv"
        path.write_text(
            "内容,一级分类,二级分类,来源\n"
            "建议增加夜间模式,建议,界面,社区\n"
            "客服态度太差,吐槽,客服,商城\n",
            encoding="utf-8-sig",
        )
        rows = load_examples(path)
        assert rows[0]["text"] == "建议增加夜间模式"
        assert rows[0]["category"] == "作品建议"
        assert rows[0]["sub_category"] == "界面优化"
        assert rows[1]["category"] == "舆情吐槽"
        assert rows[1]["sub_category"] == "服务态度"
        assert rows[1]["source"] == "商城"

    def test_explicit_field_mapping(self, tmp_path):
        from src.classifier.dataset import load_examples

        path = tmp_path / "custom.jsonl"
        path.write_text(
            json.dumps({"body": "APP闪退", "l1": "售后报修", "l2": "软件缺陷"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        rows = load_examples(path, text_field="body", category_field="l1", sub_category_field="l2")
        assert rows == [{"text": "APP闪退", "category": "售后报修",
                         "sub_category": "软件缺陷", "source": "社区"}]
