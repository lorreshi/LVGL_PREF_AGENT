"""OpenAIProvider —— 基于 langchain-openai 的 ``ChatOpenAI`` 后端。

对照 design.md "LLM_Provider" 章节与需求 12.3：以 ``type: openai`` 经工厂构造。
读取 ``LLMConfig`` 的 ``model`` / ``temperature`` / ``base_url``（如指定则用于
自定义端点），并从 ``api_key_env`` 命名的环境变量取密钥。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from embedded_device_agent.core.config.models import LLMConfig
from embedded_device_agent.core.llm.base import BaseLLMProvider
from embedded_device_agent.core.llm.factory import LLMProviderFactory
from embedded_device_agent.core.llm.providers._common import require_api_key

__all__ = ["OpenAIProvider"]


@LLMProviderFactory.register("openai")
class OpenAIProvider(BaseLLMProvider):
    """封装 ``ChatOpenAI`` 的 LLM_Provider。"""

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "openai"

    def get_chat_model(self) -> BaseChatModel:
        """构造并返回 ``ChatOpenAI``；密钥在此时才从环境变量读取。"""
        api_key = require_api_key(self._cfg.api_key_env)
        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "api_key": api_key,
            "temperature": self._cfg.temperature,
        }
        if self._cfg.base_url:
            kwargs["base_url"] = self._cfg.base_url
        return ChatOpenAI(**kwargs)
