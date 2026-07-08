"""任务 15.3：Capability B（FunctionalTest）骨架测试。

在 Test_Harness 上以 ``FakeDeviceIO`` 跑通最小功能测试流程，证明平台契约
（``BaseCapability`` + ``CapabilityFactory`` + subgraph-as-tool）对**第二个能力**成立
（Property 10），且共享 DeviceIO 被本能力复用（需求 18.2）。

**Validates: Requirements 19.1, 19.4, 18.2**
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from embedded_device_agent.capabilities.factory import CapabilityFactory
from embedded_device_agent.capabilities.functional_test.capability import (
    FunctionalTestCapability,
)
from embedded_device_agent.capabilities.functional_test.graph import (
    build_functional_test_graph,
)
from embedded_device_agent.capabilities.functional_test.models import (
    Assertion,
    TestCase,
    TestStep,
)
from embedded_device_agent.core.config.models import (
    AppConfig,
    DeviceConfig,
    LLMConfig,
    RetrieverConfig,
)
from embedded_device_agent.core.memory.backends.local_store import LocalStoreRetriever
from embedded_device_agent.core.services import CoreServices
from tests.harness import FakeDeviceIO, FakeLLMProvider

import embedded_device_agent.capabilities  # noqa: F401 - 触发注册


def _core(device: FakeDeviceIO) -> CoreServices:
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
        device=device,
        llm=FakeLLMProvider(responses=["ok"]),
        retriever=LocalStoreRetriever(cfg.retriever, store=store),
        checkpointer=InMemorySaver(),
        store=store,
        config=cfg,
    )


def _ping_case() -> TestCase:
    return TestCase(
        name="ping-pong",
        steps=[
            TestStep(
                name="inject-touch",
                action="inject_input",
                input_kind="touch",
                input_params={"x": 10, "y": 20},
            ),
            TestStep(
                name="ping",
                action="send_cmd",
                command="PING",
                assertion=Assertion(op="contains", expected="PONG"),
            ),
        ],
    )


def test_capability_registered_and_is_functional_test() -> None:
    """Capability B 经导入自动注册，且可由工厂构造（Property 10）。"""
    assert "functional_test" in CapabilityFactory.available()
    core = _core(FakeDeviceIO())
    cap = CapabilityFactory.create("functional_test", core)
    assert isinstance(cap, FunctionalTestCapability)
    assert cap.name == "functional_test"
    assert cap.description


def test_minimal_flow_runs_end_to_end_via_shared_deviceio() -> None:
    """最小功能测试流程经共享 DeviceIO 端到端跑通并产出报告（需求 19.1/19.4/18.2）。"""
    device = FakeDeviceIO(cmd_responses=["PONG"])
    core = _core(device)
    graph = build_functional_test_graph(core)

    final = graph.invoke(
        {"case": _ping_case()}, config={"configurable": {"thread_id": "ft-1"}}
    )

    assert final["done"] is True
    report = final["report"]
    assert report.passed is True
    assert report.total_steps == 2
    assert report.passed_steps == 2
    # 共享 DeviceIO 被复用：注入输入与发命令均经同一 FakeDeviceIO（需求 18.2）。
    assert device.injected_inputs and device.injected_inputs[0].kind == "touch"
    assert ("send_cmd", "PING") in device.calls
    assert device.opened == ("/dev/ttyFAKE0", 921600)


def test_failing_assertion_marks_report_failed() -> None:
    """断言不满足时步骤与整例判为失败（需求 19.4）。"""
    device = FakeDeviceIO(cmd_responses=["ERR"])
    core = _core(device)
    graph = build_functional_test_graph(core)

    final = graph.invoke(
        {"case": _ping_case()}, config={"configurable": {"thread_id": "ft-2"}}
    )
    report = final["report"]
    assert report.passed is False
    assert report.failed_steps == 1


def test_capability_as_tool_runs_case_and_returns_report() -> None:
    """as_tool 暴露的 subgraph-as-tool 可运行用例并回传语言中立报告（需求 16.4）。"""
    device = FakeDeviceIO(cmd_responses=["PONG"])
    core = _core(device)
    cap = CapabilityFactory.create("functional_test", core)
    ft_tool = cap.as_tool(core)

    result = ft_tool.invoke(
        {
            "type": "tool_call",
            "name": "functional_test",
            "id": "call-1",
            "args": {"case": _ping_case().model_dump(mode="json")},
        }
    )
    # tool 返回 ToolMessage，其 content 承载报告摘要（JSON 可序列化）。
    content = getattr(result, "content", result)
    assert "ping-pong" in str(content)
