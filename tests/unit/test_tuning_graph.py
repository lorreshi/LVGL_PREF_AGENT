"""任务 12.5：调优闭环 subgraph 单元测试（需求 6.5 / 11.4 / 11.5）。

用 ``FakeDeviceIO`` + ``FakeLLMProvider``（+ 注入的 fake 子智能体）完整离线驱动
``build_tuning_graph`` 构建的闭环图，覆盖：

* **decide 分支**：无慢帧 → memorize 达标；有慢帧且未达标 → optimize 复测。
* **迭代计数**：``decide`` 每轮 ``iteration++``。
* **halt 触发**：永不达标时在 ``max_iterations`` 处 halt 并收尾。
* **interrupt / resume**：缺场景在 intake 处 ``interrupt``，resume 后 Checkpointer
  恢复状态继续；Half_Auto 在 ``collect`` 干预点 ``interrupt``。
* **Property 5：迭代有界终止** —— 闭环迭代不超过 ``max_iterations``，有限步内到达
  memorize/halt 终态。**Validates: Requirements 6.5, 11.4, 11.5**

不触达真实硬件 / LLM / 网络：设备与 LLM 经 harness fake 替换，Store 用
``InMemoryStore``；采集回放 ``tests/fixtures/recorded_capture.log``（3 帧，最慢帧
38000us）。
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from embedded_device_agent.capabilities.perf_tuning.agents.optimizer import (
    AppliedOptimization,
    OptimizerResult,
)
from embedded_device_agent.capabilities.perf_tuning.graph import build_tuning_graph
from embedded_device_agent.core.config.models import (
    AppConfig,
    DeviceConfig,
    LLMConfig,
    RetrieverConfig,
)
from embedded_device_agent.core.memory.backends.local_store import LocalStoreRetriever
from embedded_device_agent.core.models import KnowledgeRecord
from embedded_device_agent.core.services import CoreServices
from tests.harness import FakeDeviceIO, FakeLLMProvider

# recursion_limit 需足够容纳多轮闭环（每轮约 5 节点 + 首尾节点）。
_RUN_CONFIG = {"recursion_limit": 500}


# --------------------------------------------------------------------------- #
# 脚手架
# --------------------------------------------------------------------------- #
def _config(
    *,
    frame_budget_us: int,
    max_iterations: int = 3,
    mode: str = "full_auto",
    intervention_points: list[str] | None = None,
) -> AppConfig:
    return AppConfig(
        llm=LLMConfig(type="anthropic", model="m", api_key_env="ANTHROPIC_API_KEY"),
        device=DeviceConfig(type="fake", port="/dev/ttyFAKE0"),
        retriever=RetrieverConfig(),
        frame_budget_us=frame_budget_us,
        hotspot_top_n=20,
        report_token_budget=0,
        mode=mode,  # type: ignore[arg-type]
        max_iterations=max_iterations,
        intervention_points=intervention_points or [],
    )


def _services(cfg: AppConfig) -> CoreServices:
    store = InMemoryStore()
    return CoreServices(
        device=FakeDeviceIO(),
        llm=FakeLLMProvider(responses=["ok"]),
        retriever=LocalStoreRetriever(cfg.retriever, store=store),
        checkpointer=InMemorySaver(),
        store=store,
        config=cfg,
    )


class _FakeOptimizer:
    """注入用的 fake Optimizer：full_auto 恒"应用"一项优化但从不真正改善性能。

    用于「永不达标 → halt」与「无改进 → 标记无效回退」路径，避免触达真实 LLM 的
    结构化输出（harness FakeLLMProvider 不支持 with_structured_output）。
    """

    def __init__(self) -> None:
        self.calls = 0

    def run(self, hotspot, mode, *, report=None, context: str = ""):  # noqa: ANN001
        self.calls += 1
        if hotspot is None:
            return OptimizerResult(status="no_hotspot")
        return OptimizerResult(
            status="applied",
            applied=AppliedOptimization(
                target_func=hotspot.func,
                file_path="src/fake.c",
                summary=f"fake 优化 {hotspot.func}",
            ),
        )


def _thread_config(thread_id: str) -> dict:
    return {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}


# --------------------------------------------------------------------------- #
# decide 分支：无慢帧 → memorize 达标（happy path，全程离线）
# --------------------------------------------------------------------------- #
def test_no_slow_frames_reaches_memorize_and_persists_knowledge() -> None:
    # 帧预算极高 → 录制日志无慢帧 → decide 直接走 memorize（达标）。
    cfg = _config(frame_budget_us=1_000_000, max_iterations=3)
    services = _services(cfg)
    graph = build_tuning_graph(services)

    final = graph.invoke(
        {"scenario": "滚动列表", "mode": "full_auto"},
        config=_thread_config("t-happy"),
    )

    assert final["done"] is True
    assert final["latest"].no_slow_frames is True
    # 仅一轮 decide（iteration=1），未进入 optimize/instrument。
    assert final["iteration"] == 1
    assert final.get("optimizations_applied") == []
    # 达标时沉淀一条知识（需求 9.1）——跨线程 Store 可召回。
    recalled = services.retriever.recall("滚动列表", 5)
    assert any(isinstance(r, KnowledgeRecord) for r in recalled)


# --------------------------------------------------------------------------- #
# decide 分支 + 迭代计数 + halt：永不达标 → 在 max_iterations 处 halt
# --------------------------------------------------------------------------- #
def test_never_improving_halts_at_max_iterations() -> None:
    # 帧预算极低 → 恒有慢帧；fake 优化器"应用"但复测仍不达标 → 迭代至上限 halt。
    cfg = _config(frame_budget_us=1, max_iterations=2)
    services = _services(cfg)
    fake_opt = _FakeOptimizer()
    graph = build_tuning_graph(services, optimizer=fake_opt)

    final = graph.invoke(
        {"scenario": "永不达标场景", "mode": "full_auto"},
        config=_thread_config("t-halt"),
    )

    assert final["done"] is True
    # 迭代有界：不超过 max_iterations（Property 5）。
    assert final["iteration"] == 2
    # 每轮都尝试优化 → 应用记录累计；无改进者被标记无效回退（需求 8.3）。
    assert len(final.get("optimizations_applied", [])) >= 1
    assert len(final.get("invalidated_optimizations", [])) >= 1
    assert final.get("last_verified_effective") is False
    assert fake_opt.calls >= 1


# --------------------------------------------------------------------------- #
# interrupt / resume：缺场景在 intake 处暂停，resume 后恢复继续
# --------------------------------------------------------------------------- #
def test_missing_scenario_interrupts_then_resumes() -> None:
    cfg = _config(frame_budget_us=1_000_000, max_iterations=3)
    services = _services(cfg)
    graph = build_tuning_graph(services)
    config = _thread_config("t-interrupt")

    # 缺场景 → intake 处 interrupt 暂停（图未完成，状态由 Checkpointer 保留）。
    first = graph.invoke({"scenario": "", "mode": "full_auto"}, config=config)
    assert "__interrupt__" in first

    # resume 提供场景 → 恢复继续直至完成。
    final = graph.invoke(Command(resume="补充的场景"), config=config)
    assert final["done"] is True
    assert final["scenario"] == "补充的场景"


# --------------------------------------------------------------------------- #
# Half_Auto 人在环：collect 干预点 interrupt，resume 后继续
# --------------------------------------------------------------------------- #
def test_half_auto_collect_intervention_interrupts_then_resumes() -> None:
    cfg = _config(
        frame_budget_us=1_000_000,
        max_iterations=3,
        mode="half_auto",
        intervention_points=["collect"],
    )
    services = _services(cfg)
    graph = build_tuning_graph(services)
    config = _thread_config("t-halfauto")

    first = graph.invoke(
        {"scenario": "手动触发场景", "mode": "half_auto"}, config=config
    )
    assert "__interrupt__" in first  # 在 collect 处等待人工触发场景

    final = graph.invoke(Command(resume="triggered"), config=config)
    assert final["done"] is True


# --------------------------------------------------------------------------- #
# Property 5：迭代有界终止（hypothesis）
# **Validates: Requirements 6.5, 11.4, 11.5**
# --------------------------------------------------------------------------- #
@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(max_iters=st.integers(min_value=1, max_value=4))
def test_property5_bounded_termination(max_iters: int) -> None:
    """永不达标场景下，闭环必在有限步到达终态且 iteration 不超过 max_iterations。"""
    cfg = _config(frame_budget_us=1, max_iterations=max_iters)
    services = _services(cfg)
    graph = build_tuning_graph(services, optimizer=_FakeOptimizer())

    final = graph.invoke(
        {"scenario": f"永不达标-{max_iters}", "mode": "full_auto"},
        config=_thread_config(f"t-prop5-{max_iters}"),
    )

    assert final["done"] is True
    assert 1 <= final["iteration"] <= max_iters
