"""Memory / Retriever 底座子包。

长期记忆的可插拔抽象：``BaseRetriever`` 声明语义召回（``recall``）与知识
持久化（``persist``）契约，``RetrieverFactory`` 依 ``RetrieverConfig.type``
分发构造具体后端（``local_store`` / ``external_rag``，任务 5.2 实现）。

风格与 LLM / Device 两套底座（任务 3.1 / 4.1）保持一致：所有记忆副作用都
藏在基类 + 工厂之后，测试时以内存实现替换即可脱离真实 Store / 外部 RAG。
"""

from __future__ import annotations

from embedded_device_agent.core.memory.base import BaseRetriever
from embedded_device_agent.core.memory.factory import RetrieverFactory

# 导入具体后端子包以触发 ``@RetrieverFactory.register(...)`` 注册（任务 5.2）。
# 放在契约/工厂导入之后以避免循环导入。
from embedded_device_agent.core.memory import backends as backends  # noqa: E402,F401

__all__ = ["BaseRetriever", "RetrieverFactory"]
