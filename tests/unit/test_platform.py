"""任务 14.4：平台层（BaseCapability / CapabilityFactory / Router）单元测试。

对照需求 16.3/16.4/17.1/17.2/17.3 与 **Property 10：能力注册即可路由**——任一遵循
``BaseCapability`` 契约并经 ``CapabilityFactory`` 注册的能力，无需修改 Router/核心代码
即被纳入分发候选。

不触达真实硬件 / LLM：以 harness fake 构造 CoreServices，在工具装配 / 候选枚举层面
断言路由候选（Router 主体经 ``create_react_agent`` 绑定工具需 tool-calling 模型，
故 Property 10 于候选层验证）。

**Validates: Requirements 16.3, 16.4, 17.1, 17.2, 17.3**
"""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from embedded_device_agent.capabilities.base import BaseCapability
from embedded_device_agent.capabilities.factory import CapabilityFactory
from embedded_device_agent.core.config.models import (
    AppConfig,
    DeviceConfig,
    LLMConfig,
    RetrieverConfig,
)
from embedded_device_agent.core.memory.backends.local_store import LocalStoreRetriever
from embedded_device_agent.core.services import CoreServices
from embedded_device_agent.router import (
    available_capabilities,
    build_capability_tools,
)
from tests.harness import FakeDeviceIO, FakeLLMProvider

# 触发内置能力注册（perf_tuning / functional_test）。
import embedded_device_agent.capabilities  # noqa: F401


def _core() -> CoreServices:
    store = InMemoryStore()
    cfg = AppConfig(
        llm=LLMConfig(type="anthropic", model="m", api_key_env="ANTHROPIC_API_KEY"),
        device=DeviceConfig(type="fake", port="/dev/ttyFAKE0"),
        retriever=RetrieverConfig(),
        frame_budget_us=16667,
        mode="full_auto",
        max_iterations=3,
    )
    return CoreServices(
        device=FakeDeviceIO(),
        llm=FakeLLMProvider(responses=["ok"]),
        retriever=LocalStoreRetriever(cfg.retriever, store=store),
        checkpointer=InMemorySaver(),
        store=store,
        config=cfg,
    )


# --------------------------------------------------------------------------- #
# CapabilityFactory：注册 / 构造 / 未知键报错
# --------------------------------------------------------------------------- #
def test_builtin_capabilities_registered() -> None:
    """内置两个能力经导入自动注册（需求 16.2）。"""
    available = CapabilityFactory.available()
    assert "perf_tuning" in available
    assert "functional_test" in available


def test_create_unknown_capability_lists_available() -> None:
    """未知能力键报清晰错误并列出可用能力（需求 17.2 的精神）。"""
    with pytest.raises(ValueError) as exc:
        CapabilityFactory.create("does_not_exist", _core())
    msg = str(exc.value)
    assert "does_not_exist" in msg
    assert "perf_tuning" in msg


def test_factory_rejects_non_capability_and_duplicate() -> None:
    with pytest.raises(TypeError):

        @CapabilityFactory.register("bad_platform_cap")
        class NotACapability:  # 非 BaseCapability 子类
            pass

    with pytest.raises(ValueError):

        @CapabilityFactory.register("perf_tuning")  # 已注册
        class Dup(BaseCapability):
            name = "perf_tuning"
            description = "dup"

            def build_graph(self, core):  # noqa: ANN001
                raise NotImplementedError

            def as_tool(self, core):  # noqa: ANN001
                raise NotImplementedError


# --------------------------------------------------------------------------- #
# Property 10：注册即可路由——新增能力无需改 Router/核心即进入候选与工具集
# --------------------------------------------------------------------------- #
@pytest.fixture
def _temp_capability_key():
    """注册一个临时能力用于验证扩展性，测试后从注册表移除以免串扰。"""
    key = "dummy_probe_cap"

    @CapabilityFactory.register(key)
    class DummyProbeCapability(BaseCapability):
        name = key
        description = "临时探针能力：用于验证注册即可路由（Property 10）。"

        def build_graph(self, core: CoreServices) -> CompiledStateGraph:  # pragma: no cover
            raise NotImplementedError

        def as_tool(self, core: CoreServices) -> BaseTool:
            @tool(key)
            def _probe(query: str) -> str:
                """探针能力工具。"""
                return f"probe:{query}"

            return _probe

    try:
        yield key
    finally:
        CapabilityFactory._registry.pop(key, None)


def test_property10_newly_registered_capability_is_routable(_temp_capability_key) -> None:
    """新注册能力**无需改动 Router** 即出现在候选枚举与工具装配中（Property 10）。"""
    key = _temp_capability_key

    # 候选枚举：注册即入选（需求 17.1/17.2）。
    candidates = {c["key"] for c in available_capabilities()}
    assert key in candidates

    # 工具装配：注册即被 Router 装配为可调用工具（需求 16.4）。
    tools = build_capability_tools(_core())
    tool_names = {t.name for t in tools}
    assert key in tool_names
    assert "perf_tuning" in tool_names
    assert "functional_test" in tool_names


def test_available_capabilities_carry_description_for_intent_dispatch() -> None:
    """每个候选携带用于意图分发的自然语言描述（需求 17.1）。"""
    for cap in available_capabilities():
        assert cap["description"], f"能力 {cap['key']} 缺少 description"


def test_capability_tools_are_base_tools() -> None:
    """装配出的候选均为 LangChain BaseTool（subgraph-as-tool，需求 16.4）。"""
    tools = build_capability_tools(_core())
    assert tools and all(isinstance(t, BaseTool) for t in tools)
