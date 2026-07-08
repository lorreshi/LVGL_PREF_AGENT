"""LLM_Provider 基类。

严格对照 design.md "Components and Interfaces → 基础设施层 → LLM_Provider"：

``BaseLLMProvider`` 封装一个可供 agent 使用的 chat model，向编排层暴露统一契约，
使"换模型不改核心"成立——所有 LLM 副作用都藏在该基类 + 工厂之后，由 YAML 的
``type`` 字段选择具体实现（需求 12.3）。

具体后端实现（AnthropicProvider / OpenAIProvider / OpenAICompatibleProvider）在
任务 3.2 于 ``providers/`` 子包中提供，并注册到 ``LLMProviderFactory``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel

__all__ = ["BaseLLMProvider"]


class BaseLLMProvider(ABC):
    """封装一个可供 agent 使用的 chat model 的抽象契约。

    子类通常在 ``__init__`` 中接收一个 ``LLMConfig``，并据此惰性构造底层
    LangChain chat model。副作用（真实 API 客户端、密钥读取）应发生在具体
    子类内部，基类只声明契约。
    """

    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        """返回一个可供 LangGraph/agent 使用的 LangChain ``BaseChatModel`` 实例。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """该 provider 的可读标识（如 ``"anthropic"``、``"openai"``）。"""
        raise NotImplementedError
