"""BERT 小模型接口、持久化与配置分发测试（使用本地微型主干，不依赖网络）。"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture()
def tiny_backbone(tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizer

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab += list("闪退崩溃小米手机系统卡顿客服物流价格质量建议界面推荐售后报修")
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(dict.fromkeys(vocab)), encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=False)
    config = BertConfig(
        vocab_size=len(set(vocab)), hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
    )
    return BertModel(config), tokenizer


def _build_model(tiny_backbone):
    from src.classifier.bert import CATEGORIES, SUB_CATEGORIES, BertSmallModel, _Heads

    backbone, tokenizer = tiny_backbone
    heads = _Heads(backbone.config.hidden_size, len(CATEGORIES), len(SUB_CATEGORIES))
    return BertSmallModel(backbone, tokenizer, heads, max_len=32, backbone_name="tiny")


class TestBertSmallModel:
    def test_predict_contract(self, tiny_backbone):
        from src.classifier.bert import CATEGORIES, SUB_CATEGORIES
        from src.classifier.base import SmallModelPrediction

        model = _build_model(tiny_backbone)
        pred = model.predict("小米手机老是闪退崩溃")
        assert isinstance(pred, SmallModelPrediction)
        assert pred.category in CATEGORIES
        assert pred.sub_category in SUB_CATEGORIES
        assert 0.0 <= pred.category_confidence <= 1.0
        assert 0.0 <= pred.sub_confidence <= 1.0
        assert set(pred.category_probs) == set(CATEGORIES)
        assert abs(sum(pred.category_probs.values()) - 1.0) < 0.01
        assert model.predict("   ") is None

    def test_save_and_load_roundtrip(self, tiny_backbone, tmp_path):
        model = _build_model(tiny_backbone)
        before = model.predict("客服态度太差了")
        model.save(tmp_path / "bert")

        from src.classifier.bert import BertSmallModel
        loaded = BertSmallModel.load(tmp_path / "bert")
        after = loaded.predict("客服态度太差了")

        assert after.category == before.category
        assert after.sub_category == before.sub_category
        assert abs(after.category_confidence - before.category_confidence) < 1e-4
        assert json.loads((tmp_path / "bert" / "config.json").read_text(encoding="utf-8"))["max_len"] == 32

    def test_config_dispatches_to_bert(self, tiny_backbone, tmp_path, monkeypatch):
        from src import tools
        from src.classifier.bert import BertSmallModel
        from src.config import get_settings

        model = _build_model(tiny_backbone)
        model.save(tmp_path / "bert")
        cfg = get_settings()
        monkeypatch.setattr(cfg, "small_model_type", "bert")
        monkeypatch.setattr(cfg, "small_model_path", str(tmp_path / "bert"))
        monkeypatch.setattr(tools, "_small_model", None)
        monkeypatch.setattr(tools, "_small_model_failed", False)

        loaded = tools._get_small_model()
        assert isinstance(loaded, BertSmallModel)
        assert tools.small_model_predict("物流太慢了") is not None
