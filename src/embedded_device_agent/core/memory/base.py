"""Retriever（长期记忆）基类。

严格对照 design.md "Components and Interfaces → 基础设施层 → Retriever"：

``BaseRetriever`` 是对**长期记忆检索面**的抽象——语义召回历史知识
（``recall``）与持久化新知识（``persist``），使跨线程知识复用与"可插外部
RAG"成立（需求 9.4 / 12.5）。所有记忆副作用都藏在该基类 + 工厂之后，由
YAML 的 ``type`` 字段选择具体实现；测试时以内存实现替换即可脱离真实
LangGraph Store / 外部 RAG。

具体后端实现（LocalStoreRetriever / ExternalRAGRetriever）在任务 5.2 于
``backends/`` 子包中提供，并经 ``@RetrieverFactory.register(...)`` 注册。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from embedded_device_agent.core.models import KnowledgeRecord

__all__ = ["BaseRetriever"]


class BaseRetriever(ABC):
    """长期记忆检索面的统一抽象契约（需求 9.4）。

    子类通常在 ``__init__`` 中接收一个 ``RetrieverConfig``（以及本地后端所需的
    Store 句柄），据此持有底层检索资源。真实副作用发生在具体子类内部，基类只
    声明契约。
    """

    @abstractmethod
    def recall(self, symptom: str, k: int) -> list[KnowledgeRecord]:
        """按症状语义召回至多 ``k`` 条历史 ``KnowledgeRecord``，按相关性排序。

        无相关知识时返回空列表。跨线程可见（需求 9.5）由具体后端保证。
        """
        raise NotImplementedError

    @abstractmethod
    def persist(self, record: KnowledgeRecord) -> str:
        """持久化一条 ``KnowledgeRecord`` 并返回其存储标识（id）。"""
        raise NotImplementedError
