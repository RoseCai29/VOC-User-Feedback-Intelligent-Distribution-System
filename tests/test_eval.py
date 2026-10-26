from src.evaluate import evaluate, label_rules, load_dataset, REAL_EVAL_SET_PATH
from src.state import FeedbackCategory, FeedbackSubCategory


_CATS = {c.value for c in FeedbackCategory}
_SUBS = {s.value for s in FeedbackSubCategory}


class TestEvalSet:
    def test_labels_valid(self):
        # 验收口径：真实人工标注 test 集（合成集已退出默认）
        examples = load_dataset(REAL_EVAL_SET_PATH)
        assert len(examples) >= 100, "真实 test 集样本过少"
        texts = [e["text"] for e in examples]
        assert len(texts) == len(set(texts)), "存在重复样例"
        for e in examples:
            assert e["text"].strip(), f"空文本样例: {e}"
            assert e["category"] in _CATS, f"非法一级分类: {e}"
            assert e["sub_category"] in _SUBS, f"非法二级分类: {e}"

    def test_rules_baseline(self):
        # 规则基线守门：真实 test 集实测 L1 74.5% / L2 66.7% / 宏F1 0.73，
        # 断言下限贴着实测值留 ~0.5pt 容差，任何改动导致基线退化即 CI 失败
        examples = load_dataset(REAL_EVAL_SET_PATH)
        result = evaluate(label_rules, examples)
        assert result["total"] == len(examples)
        assert result["level1_accuracy"] >= 0.74, result
        assert result["level2_accuracy"] >= 0.66, result
        assert result["macro_f1"] >= 0.70, result
        assert result["micro_f1"] == result["level1_accuracy"], "单标签多分类下微F1恒等于准确率"
