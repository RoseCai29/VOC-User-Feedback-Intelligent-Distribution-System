"""分类器能力层：小模型接口 + 置信度级联路由。

注意：此处不导入 linear，避免 sklearn 缺失时影响主链路启动。
"""

from .base import SmallModel, SmallModelPrediction
from .router import CascadeRouter

__all__ = ["SmallModel", "SmallModelPrediction", "CascadeRouter"]
