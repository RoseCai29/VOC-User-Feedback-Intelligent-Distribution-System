"""
配置中心 —— pydantic-settings 统一管理所有环境变量和默认值
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings

# 以项目根目录为基准，避免从其他工作目录启动时读不到 .env
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    openai_api_key: str = "sk-placeholder"
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: float = 30.0
    use_deterministic: bool = False
    max_retries: int = 3
    max_loop_count: int = 20
    confidence_threshold: float = 0.7
    dedup_threshold: float = 0.75
    # 分类通道: rules(纯规则) | semantic(纯LLM) | small(纯小模型) | cascade(小模型优先+LLM兜底)
    # 默认 rules：真实 test 集实测 cascade L1 61.4% < rules 74.5%（小模型域偏移，见 README Known
    # Limitations）。待真实标注重训 + 阈值重标定达标后再切回 cascade。
    classifier_mode: str = "rules"
    cascade_fallback: str = "llm"                     # 小模型低置信时的兜底: llm | rules
    small_model_type: str = "bert"                    # bert | linear
    small_model_path: str = "./data/models/bert_voc_v2"  # BERT 模型目录(真实标注重训版; 旧伪标签版为 bert_voc)
    small_linear_model_path: str = "./data/models/small_clf.joblib"  # 线性模型基线
    small_model_category_threshold: float = 0.80      # 一级直出阈值(真实 test 验证: 直出 L1 85.2%)
    small_model_sub_threshold: float = 0.75           # 二级直出阈值(真实 test 验证: 直出 L2 75.3%)
    # fast path（仅 LLM 模式触发）：真实 test 实测双阈值直出 L1 85.2% / L2 75.3%（覆盖 52.9%），
    # 高于规则基线 74.5%/66.7%，经真实标注重训(bert_voc_v2)后开启；
    # 旧伪标签模型同阈值仅 53.5%，历史教训见 README Known Limitations。
    enable_fast_path: bool = True
    enable_pending_pool: bool = True                  # 低置信样本写入待标注池
    bootstrap_min_confidence: float = 0.8             # 伪标签进入训练集的最低LLM置信度
    # 生产护栏
    max_input_length: int = 2000          # 单条反馈最大字符数
    max_trace_steps: int = 60             # trace 轨迹有界上限
    dedup_max_bucket_members: int = 500   # 去重桶成员有界上限
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "voc_evidence"
    embedding_model: str = "text-embedding-3-small"
    evidence_top_k: int = 3
    crawl_default_url: str = "https://www.xiaomi.cn/community/feedbacks"
    crawl_max_items: int = 10
    crawl_timeout: float = 30.0
    database_url: str = "sqlite:///./data/voc.db"
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    log_level: str = "INFO"
    data_dir: str = "./data"
    log_dir: str = "./logs"
    enable_live_crawl: bool = False
    crawl_user_agent: str = "Mozilla/5.0 (VOC-Agent/1.0)"
    cors_origins: str = "*"

    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        os.makedirs(_settings.data_dir, exist_ok=True)
        os.makedirs(_settings.log_dir, exist_ok=True)
    return _settings