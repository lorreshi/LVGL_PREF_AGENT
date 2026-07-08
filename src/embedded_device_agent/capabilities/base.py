"""可插拔能力的统一契约 ``BaseCapability``（任务 14.1）。

严格对照 design.md「平台层 → BaseCapability 与 CapabilityFactory」与需求 16：

每个能力都是遵循 ``BaseCapability`` 契约、自带 subgraph/子智能体/工具/状态的独立
模块。平台顶层 ``Router_Orchestrator`` 依各能力的 ``description`` 做意图分发，并把每个
能力当作**工具（subgraph-as-tool）**调用（需求 16.4/17.1）。新增能力只需实现该契约并
经 ``CapabilityFactory`` 注册，**零改动路由与底座**（需求 16.3，Property 10）。

* ``name`` / ``description``：能力标识与供 Router 意图匹配的自然语言描述
  （需求 16.1/17.1）。
* ``build_graph(core)``：用注入的共享底座 :class:`CoreServices` 构建并返回该能力的
  已编译 LangGraph subgraph（需求 16.4；副作用经底座注入，测试可替换 fake）。
* ``as_tool(core)``：把该能力的 subgraph 暴露为一个 LangChain 工具，供 Router 装配
  为可调用工具（subgraph-as-tool，需求 16.4）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from embedded_device_agent.core.services import CoreServices

__all__ = ["BaseCapability"]


class BaseCapability(ABC):
    """可插拔能力契约：构建 subgraph 并暴露为 subgraph-as-tool（需求 16）。

    子类需提供类级 ``name`` / ``description``，并实现 :meth:`build_graph` 与
    :meth:`as_tool`。``description`` 应清晰陈述能力职责与适用场景，供 Router 做意图
    分发（需求 17.1）。
    """

    #: 能力唯一标识（亦为 ``CapabilityFactory`` 注册键）。
    name: str = ""
    #: 供 Router 意图匹配的自然语言描述（需求 16.1/17.1）。
    description: str = ""

    @abstractmethod
    def build_graph(self, core: CoreServices) -> CompiledStateGraph:
        """用注入的共享底座构建并返回该能力已编译的 LangGraph subgraph（需求 16.4）。"""
        raise NotImplementedError

    @abstractmethod
    def as_tool(self, core: CoreServices) -> BaseTool:
        """把该能力暴露为一个可被 Router 装配调用的工具（subgraph-as-tool，需求 16.4）。"""
        raise NotImplementedError
