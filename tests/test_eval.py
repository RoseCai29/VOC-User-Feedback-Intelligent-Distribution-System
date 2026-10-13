from src.evaluate import load_eval_set, evaluate, label_rules
from src.state import FeedbackCategory, FeedbackSubCategory


_CATS = {c.value for c in FeedbackCategory}
_SUBS = {s.value for s in FeedbackSubCategory}


class TestEvalSet:
    def test_labels_valid(self):
        examples = load_eval_set()
        assert len(examples) >= 15, "评估集样本过少"
        texts = [e["text"] for e in examples]
        assert len(texts) == len(set(texts)), "存在重复样例"
        for e in examples:
            assert e["text"].strip(), f"空文本样例: {e}"
            assert e["category"] in _CATS, f"非法一级分类: {e}"
            assert e["sub_category"] in _SUBS, f"非法二级分类: {e}"

    def test_rules_baseline(self):
        examples = load_eval_set()
        result = evaluate(label_rules, examples)
        assert result["total"] == len(examples)
        assert result["level1_accuracy"] >= 0.8, result
        assert result["level2_accuracy"] >= 0.7, result
        assert 0.0 <= result["macro_f1"] <= 1.0