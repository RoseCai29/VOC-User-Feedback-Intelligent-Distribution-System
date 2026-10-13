"""
OpenAI-compatible LLM 调用，统一 client 创建与 JSON 输出解析。
"""

from __future__ import annotations

import json
import logging
import os

from .config import get_settings

logger = logging.getLogger(__name__)


def json_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int = 512,
) -> dict:
    """调用 LLM 并要求返回 JSON 对象。失败抛出异常，由调用方降级。

    优先使用 response_format=json_object；对不支持的 provider 回退为普通调用后手工解析。
    """
    from openai import OpenAI

    cfg = get_settings()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", cfg.openai_api_key),
        base_url=os.getenv("OPENAI_BASE_URL", cfg.openai_base_url),
        timeout=cfg.llm_timeout,
    )
    model = os.getenv("MODEL_NAME", cfg.model_name)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": cfg.llm_temperature if temperature is None else temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
    except Exception:
        resp = client.chat.completions.create(**kwargs)

    content = (resp.choices[0].message.content or "").strip()
    content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM 返回非 JSON: {content[:200]}") from e