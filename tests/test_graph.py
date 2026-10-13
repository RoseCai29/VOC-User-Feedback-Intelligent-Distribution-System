from src.graph import run_feedback_invoke
from src.tools import reset_dedup_buckets


class TestGraphNormal:
    def test_suggestion(self):
        reset_dedup_buckets()
        state = run_feedback_invoke("小米14夜景模式偏黄，希望优化", source="社区")
        assert state.classification is not None
        assert state.route_target is not None
        assert state.finished is True
        assert len(state.trace) >= 3

    def test_after_sales(self):
        reset_dedup_buckets()
        state = run_feedback_invoke("设备无法开机充电没反应需要售后维修", source="商城")
        assert state.classification.category.value == "售后报修"
        assert state.service_guide is not None
        assert state.finished is True

    def test_sentiment(self):
        reset_dedup_buckets()
        state = run_feedback_invoke("物流等了一周还没到，客服电话打不通", source="服务频道")
        assert state.classification is not None
        assert state.finished is True


class TestLowConfidenceLoop:
    def test_self_correct(self):
        reset_dedup_buckets()
        state = run_feedback_invoke("这个APP真的不好用啊，打开总是出毛病", source="社区")
        assert state.evidence_retrieved is True
        assert state.retry_count >= 1
        assert state.classification.confidence >= 0.7
        assert state.finished is True

    def test_three_rounds(self):
        reset_dedup_buckets()
        state = run_feedback_invoke("软件最近总是用不了，不知道什么情况", source="服务频道")
        assert state.evidence_retrieved is True
        assert state.finished is True


class TestCrawl:
    def test_crawl(self):
        reset_dedup_buckets()
        state = run_feedback_invoke("", source="社区")
        assert state.crawl_completed is True
        assert len(state.crawled_feedbacks) >= 1
        assert state.raw_text != ""


class TestTrace:
    def test_accumulates(self):
        reset_dedup_buckets()
        state = run_feedback_invoke("建议加入深色模式", source="社区")
        action_types = [s.action_type.value for s in state.trace]
        assert "thought" in action_types
        assert "observation" in action_types