"""OpenAICompatibleProvider —— ``ChatOpenAI`` + 自定义 ``base_url`` 后端。

对照 design.md "LLM_Provider" 章节（"兼容内网网关"）与需求 12.3：以
``type: openai_compatible`` 经工厂构造，用于对接 OpenAI 兼容协议的自建 / 内网
网关。与 ``OpenAIProvider`` 的关键区别是 ``base_url`` 为**必填**——没有自定义
端点就应直接用 ``openai`` 后端。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from embedded_device_agent.core.config.models import LLMConfig
from embedded_device_agent.core.llm.base import BaseLLMProvider
from embedded_device_agent.core.llm.factory import LLMProviderFactory
from embedded_device_agent.core.llm.providers._common import require_api_key

__all__ = ["OpenAICompatibleProvider"]


@LLMProviderFactory.register("openai_compatible")
class OpenAICompatibleProvider(BaseLLMProvider):
    """封装指向自定义端点的 ``ChatOpenAI`` 的 LLM_Provider。"""

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "openai_compatible"

    def get_chat_model(self) -> BaseChatModel:
        """构造并返回指向 ``base_url`` 的 ``ChatOpenAI``；密钥在此时才读取。"""
        if not self._cfg.base_url:
            raise ValueError(
                "openai_compatible provider 需要在 LLMConfig.base_url 指定自定义端点；"
                "若使用官方 OpenAI 端点，请改用 type: openai。"
            )
        api_key = require_api_key(self._cfg.api_key_env)
        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "api_key": api_key,
            "base_url": self._cfg.base_url,
            "temperature": self._cfg.temperature,
        }
        return ChatOpenAI(**kwargs)
