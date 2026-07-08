"""具体 LLM Provider 后端子包。

导入本子包即触发三个后端经 ``@LLMProviderFactory.register(...)`` 注册：
``anthropic`` / ``openai`` / ``openai_compatible``（任务 3.2，需求 12.3）。
"""

from embedded_device_agent.core.llm.providers.anthropic import AnthropicProvider
from embedded_device_agent.core.llm.providers.openai import OpenAIProvider
from embedded_device_agent.core.llm.providers.openai_compatible import (
    OpenAICompatibleProvider,
)

__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
]
