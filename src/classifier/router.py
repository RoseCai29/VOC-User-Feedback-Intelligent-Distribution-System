"""
置信度级联路由 —— 小模型先判，高置信直接产出；低置信升级大模型/规则。

切换点由 category/sub 两级阈值共同决定，避免"一级对、二级错"被误放行。
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from .base import SmallModel, SmallModelPrediction

logger = logging.getLogger(__name__)

FallbackFn = Callable[[str, str, str], str]


def double_threshold_pass(
    pred: SmallModelPrediction | None,
    category_threshold: float,
    sub_threshold: float,
) -> bool:
    """双阈值判定：一级、二级置信度都达标才允许小模型直出。

    fast path 与 CascadeRouter 共用此判定，避免阈值逻辑两处维护出现口径漂移。
    """
    return (pred is not None
            and pred.category_confidence >= category_threshold
            and pred.sub_confidence >= sub_threshold)


class CascadeRouter:
    def __init__(
        self,
        model: SmallModel | None,
        category_threshold: float = 0.80,
        sub_threshold: float = 0.75,
        fallback: FallbackFn | None = None,
    ):
        self.model = model
        self.category_threshold = category_threshold
        self.sub_threshold = sub_threshold
        self.fallback = fallback

    def classify(self, raw_text: str, source: str = "社区", evidence_context: str = "") -> str:
        pred = self.model.predict(raw_text) if self.model is not None else None
        # 两级置信度都达标才允许小模型直出（与 fast path 共用同一判定）
        if double_threshold_pass(pred, self.category_threshold, self.sub_threshold):
            return self._small_result(pred)

        if self.fallback is None:
            raise RuntimeError("小模型未命中且未配置大模型兜底")

        result = json.loads(self.fallback(raw_text, source, evidence_context))
        result["escalated"] = True
        result.setdefault("engine", "llm")
        result["small_prediction"] = {
            "category": pred.category if pred else None,
            "sub_category": pred.sub_category if pred else None,
            "category_confidence": round(pred.category_confidence, 4) if pred else 0.0,
            "sub_confidence": round(pred.sub_confidence, 4) if pred else 0.0,
        }
        logger.info(
            "级联升级[%s]: 小模型=%s(%.2f/%.2f) → %s",
            raw_text[:20],
            pred.sub_category if pred else "无预测",
            pred.category_confidence if pred else 0.0,
            pred.sub_confidence if pred else 0.0,
            result.get("engine"),
        )
        return json.dumps(result, ensure_ascii=False)

    def _small_result(self, pred: SmallModelPrediction) -> str:
        result = {
            "category": pred.category,
            "sub_category": pred.sub_category,
            "tags": ["小模型"],
            "confidence": round(min(pred.category_confidence, pred.sub_confidence), 4),
            "engine": "small",
            "escalated": False,
            "small_category_confidence": round(pred.category_confidence, 4),
            "small_sub_confidence": round(pred.sub_confidence, 4),
        }
        logger.info(
            "小模型直出[%s]: %s/%s (%.2f/%.2f)",
            pred.category, pred.sub_category,
            pred.category_confidence, pred.sub_confidence,
        )
        return json.dumps(result, ensure_ascii=False)
