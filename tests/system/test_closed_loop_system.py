"""任务 18：系统级闭环测试（Tier 3）。

用录制 fixture + harness fake 驱动整张调优闭环 LangGraph 图，端到端验证系统级行为：

* 18.1 全自动闭环 happy path：走通闭环并落一条 Knowledge_Record；**Property 6：知识
  跨线程可见**（另一"线程"可召回）。**Validates: Requirements 9.5, 11.2**
* 18.2 迭代上限守卫：永不达标 fixture 在 ``Max_Iteration_Limit`` 处 halt 并报告最佳
  证据；**Property 5：迭代有界终止**。**Validates: Requirements 6.5, 11.5**
* 18.3 人在环恢复：Half_Auto 在干预点 interrupt，resume 后 Checkpointer 恢复状态继续。
  **Validates: Requirements 10.2, 10.3, 11.3**

不触达真实硬件 / LLM / 网络：设备与 LLM 经 harness fake 替换，Store 用 InMemoryStore。
"""

from __future__ import annotations

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
from embedded_device_agent.core.services import CoreServices
from tests.harness import FakeDeviceIO, FakeLLMProvider

_RUN = {"recursion_limit": 500}


def _services(cfg: AppConfig, *, store: InMemoryStore | None = None) -> CoreServices:
    store = store or InMemoryStore()
    return CoreServices(
        device=FakeDeviceIO(),
        llm=FakeLLMProvider(responses=["ok"]),
        retriever=LocalStoreRetriever(cfg.retriever, store=store),
        checkpointer=InMemorySaver(),
        store=store,
        config=cfg,
    )


def _config(**kw) -> AppConfig:
    base = dict(
        llm=LLMConfig(type="anthropic", model="m", api_key_env="ANTHROPIC_API_KEY"),
        device=DeviceConfig(type="fake", port="/dev/ttyFAKE0"),
        retriever=RetrieverConfig(),
        frame_budget_us=1_000_000,
        mode="full_auto",
        max_iterations=3,
    )
    base.update(kw)
    return AppConfig(**base)


class _NeverImprovingOptimizer:
    """恒"应用"优化但从不改善性能——用于迭代上限守卫系统测试。"""

    def run(self, hotspot, mode, *, report=None, context: str = ""):  # noqa: ANN001
        if hotspot is None:
            return OptimizerResult(status="no_hotspot")
        return OptimizerResult(
            status="applied",
            applied=AppliedOptimization(
                target_func=hotspot.func, file_path="src/x.c", summary="noop"
            ),
        )


# --------------------------------------------------------------------------- #
# 18.1：全自动闭环 happy path + Property 6（知识跨线程可见）
# --------------------------------------------------------------------------- #
def test_full_auto_closed_loop_persists_knowledge_visible_cross_thread() -> None:
    """全自动闭环走通并落一条知识；另一"线程"共享 Store 可召回（Property 6）。"""
    shared_store = InMemoryStore()
    cfg = _config(frame_budget_us=1_000_000)  # 录制 fixture 无慢帧 → 达标沉淀
    services = _services(cfg, store=shared_store)
    graph = build_tuning_graph(services)

    final = graph.invoke(
        {"scenario": "滚动列表卡顿", "mode": "full_auto"},
        config={**_RUN, "configurable": {"thread_id": "sys-happy"}},
    )
    assert final["done"] is True
    assert final["latest"] is not None

    # Property 6 / 需求 11.2：另一"线程"（独立 Checkpointer、共享 Store）可召回该知识。
    reader = LocalStoreRetriever(cfg.retriever, store=shared_store)
    recalled = reader.recall("滚动列表卡顿", 5)
    assert len(recalled) >= 1


# --------------------------------------------------------------------------- #
# 18.2：迭代上限守卫 + Property 5（迭代有界终止）
# --------------------------------------------------------------------------- #
def test_never_improving_halts_at_iteration_limit_with_best_evidence() -> None:
    """永不达标场景在 Max_Iteration_Limit 处 halt，并保留最佳证据报告（Property 5）。"""
    cfg = _config(frame_budget_us=1, max_iterations=2)  # 恒有慢帧
    services = _services(cfg)
    graph = build_tuning_graph(services, optimizer=_NeverImprovingOptimizer())

    final = graph.invoke(
        {"scenario": "永不达标", "mode": "full_auto"},
        config={**_RUN, "configurable": {"thread_id": "sys-halt"}},
    )
    assert final["done"] is True
    # 迭代有界：不超过 Max_Iteration_Limit。
    assert final["iteration"] == 2
    # 报告最佳证据：仍持有最近一次帧报告（含慢帧/热点）。
    assert final["latest"] is not None
    assert final["latest"].no_slow_frames is False
    assert final["latest"].hotspots


# --------------------------------------------------------------------------- #
# 18.3：人在环恢复（Half_Auto interrupt → resume → Checkpointer 恢复继续）
# --------------------------------------------------------------------------- #
def test_half_auto_interrupt_and_resume_recovers_state() -> None:
    """Half_Auto 在 collect 干预点 interrupt，resume 后由 Checkpointer 恢复继续至完成。"""
    cfg = _config(
        frame_budget_us=1_000_000,
        mode="half_auto",
        intervention_points=["collect"],
    )
    services = _services(cfg)
    graph = build_tuning_graph(services)
    config = {**_RUN, "configurable": {"thread_id": "sys-halfauto"}}

    first = graph.invoke({"scenario": "手动场景", "mode": "half_auto"}, config=config)
    assert "__interrupt__" in first  # 在干预点暂停，状态由 Checkpointer 保留

    # resume：从断点恢复，状态延续（scenario 保留）直至完成。
    final = graph.invoke(Command(resume="triggered"), config=config)
    assert final["done"] is True
    assert final["scenario"] == "手动场景"
