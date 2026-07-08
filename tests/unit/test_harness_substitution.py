"""任务 16.2：Test_Harness 决定性替换测试。

**Property 11：Harness 决定性替换** —— 仅经注入 ``FakeDeviceIO`` / ``FakeLLMProvider``
即可驱动任意能力的完整 subgraph，能力代码**零改动**，且不触达真实硬件 / 网络 / LLM。

本测试用两个内置能力（Capability A 调优闭环、Capability B 功能测试）各自的
subgraph 验证：注入 fake 底座后可完整跑通，且多次运行结果**确定性**一致。

**Validates: Requirements 20.3, 20.4**
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from embedded_device_agent.capabilities.functional_test.graph import (
    build_functional_test_graph,
)
from embedded_device_agent.capabilities.functional_test.models import (
    Assertion,
    TestCase,
    TestStep,
)
from embedded_device_agent.capabilities.perf_tuning.graph import build_tuning_graph
from embedded_device_agent.core.config.models import (
    AppConfig,
    DeviceConfig,
    LLMConfig,
    RetrieverConfig,
)
from embedded_device_agent.core.memory.backends.local_store import LocalStoreRetriever
from embedded_device_agent.core.services import CoreServices
from tests.harness import FakeDeviceIO, FakeLLMProvider


def _core(*, frame_budget_us: int = 1_000_000) -> CoreServices:
    """全 fake 底座：不触达真实硬件 / 网络 / LLM（需求 20.3）。"""
    store = InMemoryStore()
    cfg = AppConfig(
        llm=LLMConfig(type="anthropic", model="m", api_key_env="ANTHROPIC_API_KEY"),
        device=DeviceConfig(type="fake", port="/dev/ttyFAKE0"),
        retriever=RetrieverConfig(),
        frame_budget_us=frame_budget_us,
        mode="full_auto",
        max_iterations=3,
    )
    return CoreServices(
        device=FakeDeviceIO(cmd_responses=["PONG"]),
        llm=FakeLLMProvider(responses=["ok"]),
        retriever=LocalStoreRetriever(cfg.retriever, store=store),
        checkpointer=InMemorySaver(),
        store=store,
        config=cfg,
    )


def test_capability_a_full_subgraph_driven_by_fakes() -> None:
    """注入 fake 即可驱动 Capability A 完整调优闭环，无需改动能力代码（Property 11）。"""
    core = _core(frame_budget_us=1_000_000)  # 无慢帧 → 达标收尾
    graph = build_tuning_graph(core)
    final = graph.invoke(
        {"scenario": "滚动", "mode": "full_auto"},
        config={"recursion_limit": 500, "configurable": {"thread_id": "harness-a"}},
    )
    assert final["done"] is True
    assert final["latest"] is not None


def test_capability_b_full_subgraph_driven_by_fakes() -> None:
    """注入 fake 即可驱动 Capability B 完整功能测试骨架（Property 11）。"""
    core = _core()
    graph = build_functional_test_graph(core)
    case = TestCase(
        name="smoke",
        steps=[
            TestStep(
                name="ping",
                action="send_cmd",
                command="PING",
                assertion=Assertion(op="contains", expected="PONG"),
            )
        ],
    )
    final = graph.invoke({"case": case}, config={"configurable": {"thread_id": "harness-b"}})
    assert final["done"] is True
    assert final["report"].passed is True


def test_fake_replay_is_deterministic_across_runs() -> None:
    """相同 fake 输入多次运行产出一致结果（决定性替换，需求 20.4）。"""
    results = []
    for i in range(2):
        core = _core(frame_budget_us=1_000_000)
        graph = build_tuning_graph(core)
        final = graph.invoke(
            {"scenario": "滚动", "mode": "full_auto"},
            config={"recursion_limit": 500, "configurable": {"thread_id": f"det-{i}"}},
        )
        results.append(
            (
                final["done"],
                final["iteration"],
                final["latest"].no_slow_frames,
                final["latest"].total_frames,
                final["latest"].p95_frame_us,
            )
        )
    assert results[0] == results[1]


def test_fakes_do_not_touch_real_hardware() -> None:
    """FakeDeviceIO 仅记录调用、回放录制日志，绝不打开真实串口（需求 20.3）。"""
    device = FakeDeviceIO()
    # capture 返回指向录制 fixture 的工件，不涉及真实串口。
    artifact = device.capture(1.0)
    assert artifact.path.exists()  # 指向录制 fixture 文件
    assert device.opened is None  # 未打开任何真实串口
    assert device.calls[-1][0] == "capture"
