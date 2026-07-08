"""LLM_Provider 底座：BaseLLMProvider 契约 + LLMProviderFactory 装饰器注册工厂。

具体后端（anthropic / openai / openai_compatible）由 ``providers/`` 子包实现并
经 ``@LLMProviderFactory.register(...)`` 注册（任务 3.2）。
"""

from embedded_device_agent.core.llm.base import BaseLLMProvider
from embedded_device_agent.core.llm.factory import LLMProviderFactory

__all__ = ["BaseLLMProvider", "LLMProviderFactory"]
