"""
轻量小模型 —— TF-IDF 字符 n-gram + 逻辑回归。

设计目标:
  - 无需 GPU，可离线训练、CPU 推理；
  - 与 BERT 实现共享 SmallModel 接口，后续可无缝替换；
  - 同时输出一级/二级概率，供级联路由做置信度闸门。
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .base import SmallModelPrediction

logger = logging.getLogger(__name__)


def _build_pipeline() -> Pipeline:
    """字符级 n-gram 对中文短文本更稳健，不依赖分词器。"""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
            max_features=60000,
        )),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])


class LinearSmallModel:
    """两级文本分类小模型（一级 3 类 + 二级 9 类）。"""

    def __init__(self, category_model: Pipeline, sub_model: Pipeline, meta: dict | None = None):
        self.category_model = category_model
        self.sub_model = sub_model
        self.meta = meta or {}

    def is_ready(self) -> bool:
        return self.category_model is not None and self.sub_model is not None

    @classmethod
    def train(cls, texts: list[str], categories: list[str], sub_categories: list[str]) -> "LinearSmallModel":
        if len(texts) < 3:
            raise ValueError("训练样本过少，至少需要 3 条")
        cat_model = _build_pipeline().fit(texts, categories)
        sub_model = _build_pipeline().fit(texts, sub_categories)
        meta = {
            "train_size": len(texts),
            "categories": sorted(set(categories)),
            "sub_categories": sorted(set(sub_categories)),
        }
        logger.info("小模型训练完成: %d 条, 一级 %d 类, 二级 %d 类",
                    len(texts), len(meta["categories"]), len(meta["sub_categories"]))
        return cls(cat_model, sub_model, meta)

    def predict(self, text: str) -> SmallModelPrediction | None:
        if not text.strip() or not self.is_ready():
            return None
        try:
            cat_proba = self.category_model.predict_proba([text])[0]
            sub_proba = self.sub_model.predict_proba([text])[0]
        except Exception as e:
            logger.warning("小模型推理失败: %s", e)
            return None

        cat_classes = list(self.category_model.classes_)
        sub_classes = list(self.sub_model.classes_)
        ci = int(cat_proba.argmax())
        si = int(sub_proba.argmax())
        return SmallModelPrediction(
            category=cat_classes[ci],
            sub_category=sub_classes[si],
            category_confidence=float(cat_proba[ci]),
            sub_confidence=float(sub_proba[si]),
            category_probs={c: round(float(p), 4) for c, p in zip(cat_classes, cat_proba)},
            sub_probs={c: round(float(p), 4) for c, p in zip(sub_classes, sub_proba)},
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"category_model": self.category_model, "sub_model": self.sub_model, "meta": self.meta},
            path,
        )
        logger.info("小模型已保存: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LinearSmallModel":
        bundle = joblib.load(path)
        return cls(bundle["category_model"], bundle["sub_model"], bundle.get("meta"))
