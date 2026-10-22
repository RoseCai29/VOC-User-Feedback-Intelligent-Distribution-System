"""
小模型统一接口 —— 便于线性模型 / BERT 等实现互换。

级联架构中，小模型只需暴露 predict()，路由层不关心其内部实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SmallModelPrediction:
    """小模型两级预测结果与置信度。"""

    category: str
    sub_category: str
    category_confidence: float
    sub_confidence: float
    category_probs: dict[str, float] = field(default_factory=dict)
    sub_probs: dict[str, float] = field(default_factory=dict)


class SmallModel(Protocol):
    """小模型协议：线性模型、BERT、蒸馏模型均可实现。"""

    def is_ready(self) -> bool: ...

    def predict(self, text: str) -> SmallModelPrediction | None: ...
