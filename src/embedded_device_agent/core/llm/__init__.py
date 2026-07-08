"""LLM_Provider 底座：BaseLLMProvider 契约 + LLMProviderFactory 装饰器注册工厂。

具体后端（anthropic / openai / openai_compatible）由 ``providers/`` 子包实现并
经 ``@LLMProviderFactory.register(...)`` 注册（任务 3.2）。
"""

from embedded_device_agent.core.llm.base import BaseLLMProvider
from embedded_device_agent.core.llm.factory import LLMProviderFactory

# 导入 providers 子包以触发三个具体后端（anthropic / openai /
# openai_compatible）经装饰器注册到 LLMProviderFactory（任务 3.2，需求 12.3）。
# 放在末尾避免与 base/factory 形成循环导入。
from embedded_device_agent.core.llm import providers  # noqa: E402,F401

__all__ = ["BaseLLMProvider", "LLMProviderFactory"]
