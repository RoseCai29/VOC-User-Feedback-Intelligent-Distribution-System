"""emoji 预处理测试：只统计、不武断定性，避免把「好用😭😭😭」误判为负面。"""

from src.preprocess import analyze_emoji, strip_emoji


def test_crying_emoji_is_not_negative_by_default():
    profile = analyze_emoji("这个东西太好用了😭😭😭")

    assert profile.text == "这个东西太好用了"
    assert profile.count == 3
    assert profile.ambiguous == 3
    assert profile.strong_negative == 0
    assert profile.positive == 0
    assert profile.only_emoji is False


def test_strong_negative_and_positive_counts():
    neg = analyze_emoji("垃圾手机😡😡💢")
    assert neg.strong_negative == 3
    assert neg.text == "垃圾手机"

    pos = analyze_emoji("拍照效果不错👍👍❤️")
    assert pos.positive == 3
    assert pos.text == "拍照效果不错"


def test_pure_emoji_and_bracket_emoticon():
    pure = analyze_emoji("😭😭😭")
    assert pure.only_emoji is True
    assert pure.text == ""

    sarcastic = analyze_emoji("[微笑]好的，谢谢")
    assert sarcastic.text == "好的，谢谢"
    assert sarcastic.ambiguous == 1
    assert sarcastic.strong_negative == 0


def test_strip_falls_back_to_original_when_text_empty():
    assert strip_emoji("APP闪退😭") == "APP闪退"
    assert strip_emoji("😭😭") == "😭😭"


def test_fast_path_uses_clean_text_and_records_profile(monkeypatch):
    from src import graph
    from src.classifier.base import SmallModelPrediction
    from src.config import get_settings

    monkeypatch.setattr(get_settings(), "enable_fast_path", True)

    seen = {}

    def fake_predict(text: str):
        seen["text"] = text
        return SmallModelPrediction("售后报修", "软件缺陷", 0.95, 0.90)

    monkeypatch.setattr(graph, "small_model_predict", fake_predict)
    monkeypatch.setattr(graph, "_invoke_state", lambda state: state)

    state = graph.run_feedback_invoke("APP闪退崩溃无法打开😭😭", "社区", use_llm=True)

    assert seen["text"] == "APP闪退崩溃无法打开"
    assert state.fast_path is True
    assert state.raw_text == "APP闪退崩溃无法打开"
    assert state.original_text == "APP闪退崩溃无法打开😭😭"
    assert state.emoji_ambiguous == 2


def test_fast_path_skips_pure_emoji(monkeypatch):
    from src import graph

    def _must_not_predict(text: str):
        raise AssertionError("纯表情不应进入小模型 fast path")

    monkeypatch.setattr(graph, "small_model_predict", _must_not_predict)
    monkeypatch.setattr(graph, "_invoke_state", lambda state: state)

    state = graph.run_feedback_invoke("😭😭😭", "社区", use_llm=True)

    assert state.fast_path is False
    assert state.emoji_only is True
    assert state.raw_text == "😭😭😭"


def test_small_model_receives_text_without_emoji(monkeypatch):
    from src import tools
    from src.classifier.base import SmallModelPrediction

    seen = {}

    class _Model:
        def is_ready(self):
            return True

        def predict(self, text):
            seen["text"] = text
            return SmallModelPrediction("作品建议", "功能建议", 0.9, 0.8)

    monkeypatch.setattr(tools, "_get_small_model", lambda: _Model())
    tools.small_model_predict("希望加个分屏功能🙏🙏")

    assert seen["text"] == "希望加个分屏功能"
