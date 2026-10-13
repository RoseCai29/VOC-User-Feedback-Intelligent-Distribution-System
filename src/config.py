"""
配置中心 —— pydantic-settings 统一管理所有环境变量和默认值
"""

from __future__ import annotations

import os
from pydantic_settings import BaseSettings


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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        os.makedirs(_settings.data_dir, exist_ok=True)
        os.makedirs(_settings.log_dir, exist_ok=True)
    return _settings