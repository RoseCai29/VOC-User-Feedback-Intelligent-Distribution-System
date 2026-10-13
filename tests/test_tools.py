import json
from src.tools import classify, retrieve_evidence, dedup, route, push_service_guide, crawl_feedback, reset_dedup_buckets


class TestClassify:
    def test_software_bug(self):
        r = json.loads(classify("APP经常闪退崩溃需要售后保修"))
        assert r["category"] == "售后报修"
        assert r["sub_category"] == "软件缺陷"
        assert r["confidence"] > 0.8

    def test_suggestion(self):
        r = json.loads(classify("建议增加夜间模式"))
        assert r["category"] == "作品建议"

    def test_sentiment(self):
        r = json.loads(classify("客服态度太差了"))
        assert r["category"] == "舆情吐槽"

    def test_low_confidence(self):
        r = json.loads(classify("这个APP真的不好用啊"))
        assert r["confidence"] < 0.7

    def test_with_evidence(self):
        r = json.loads(classify("这个APP真的不好用啊", evidence_context="闪退崩溃需要保修"))
        assert r["confidence"] > 0.7
        assert "证据纠正" in r["tags"]


class TestRetrieveEvidence:
    def test_returns_evidence(self):
        r = json.loads(retrieve_evidence("闪退崩溃保修"))
        assert len(r["evidence"]) >= 1
        assert "ticket_id" in r["evidence"][0]


class TestDedup:
    def test_new_feedback(self):
        reset_dedup_buckets()
        r = json.loads(dedup("全新反馈从未见过"))
        assert r["is_duplicate"] is False

    def test_duplicate(self):
        reset_dedup_buckets()
        dedup("手机屏幕碎了需要维修")
        r = json.loads(dedup("手机屏幕碎了需要维修处理"))
        assert r["is_duplicate"] is True

    def test_persists_to_db(self):
        reset_dedup_buckets()
        dedup("持久化测试专属反馈文本零一")
        from src.db import get_all_dedup_buckets
        reps = {b.representative for b in get_all_dedup_buckets()}
        assert "持久化测试专属反馈文本零一" in reps

    def test_survives_restart(self):
        reset_dedup_buckets()
        dedup("模拟重启持久化去重样本")
        import src.tools as t
        t._dedup_cache_loaded = False  # 模拟进程重启，缓存失效
        t._dedup_cache = {}
        r = json.loads(dedup("模拟重启持久化去重样本复制"))
        assert r["is_duplicate"] is True


class TestRoute:
    def test_hardware_fault(self):
        r = json.loads(route("售后报修", "硬件故障", "社区"))
        assert r["assignee"] == "周九"
        assert r["sla_hours"] == 12

    def test_fallback(self):
        r = json.loads(route("作品建议", "功能建议", "服务频道"))
        assert r["assignee"] is not None


class TestPushServiceGuide:
    def test_hardware_fault(self):
        r = json.loads(push_service_guide("设备无法开机", "硬件故障", "社区"))
        assert r["success"] is True
        assert r["guide"]["phone"] == "400-100-5678"
        assert r["guide"]["link"].startswith("https://")

    def test_default_fallback(self):
        r = json.loads(push_service_guide("其他问题", "未知分类", "社区"))
        assert r["success"] is True
        assert r["guide"]["phone"]
        assert r["guide"]["link"]


class TestCrawlFeedback:
    def test_filter_by_product_line(self):
        r = json.loads(crawl_feedback("https://www.xiaomi.cn", product_line="手机", max_items=3))
        assert r["crawled_count"] >= 1
        for item in r["items"]:
            assert item["product_line"] == "手机"

    def test_all(self):
        r = json.loads(crawl_feedback("https://www.xiaomi.cn", max_items=100))
        assert r["crawled_count"] >= 5