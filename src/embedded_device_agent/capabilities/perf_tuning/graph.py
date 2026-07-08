"""Capability A 调优闭环 subgraph（任务 12.3）。

严格对照 design.md「编排层与智能体层 → 调优闭环状态机」的节点与转移：

    [*] → intake → recall → plan → collect → filter_parse → analyze → decide
    decide → instrument   （需更细粒度）        → collect（重编译烧录后复测）
    decide → optimize      （已定位叶子热点）    → collect（应用优化后复测/验证）
    decide → memorize      （达标 / 无慢帧）     → [*]
    decide → halt          （达到 Max_Iteration_Limit）→ memorize → [*]

关键设计（对照需求 1/6/8/9/10/11）：

* **intake**（需求 1.1/1.2）：创建 Tuning_Run 并记录场景；**缺场景则先请求**
  （用 LangGraph ``interrupt`` 暂停，等待补充场景后恢复）。
* **recall**（需求 1.3/9.3）：按场景语义召回历史知识并纳入 ``recalled_knowledge``。
* **decide**（需求 6.5/11.4/11.5）：每轮 ``iteration++`` 并对照
  ``max_iterations``；``instrument`` / ``optimize`` 之后回到 ``collect`` 复测，形成
  闭环；超限走 ``halt`` 报告最佳证据后收尾。
* **可恢复**（需求 10.1/10.2）：图以注入的 Checkpointer 编译，线程范围状态可持久化、
  可从中断点恢复。

一切副作用皆经注入的 :class:`CoreServices` 完成（设备 / LLM / Retriever /
Checkpointer / Store / Config）；四个子智能体（Collector/Analyzer/Instrumentor/
Optimizer）默认由 services 构造，亦可显式注入以便测试用 fake 完整离线驱动闭环
——节点直接调用子智能体与确定性工具，**不依赖具备 tool-calling 能力的 LLM**。

注：Half_Auto 的人在环 ``interrupt`` 与优化后基线对比/知识沉淀的细化在任务 12.4
落地；本模块提供闭环骨架、迭代守卫、full_auto 主路径与 Checkpointer 接入。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from embedded_device_agent.capabilities.perf_tuning.agents import (
    Analyzer,
    Collector,
    Instrumentor,
    Optimizer,
)
from embedded_device_agent.capabilities.perf_tuning.state import TuningRunState
from embedded_device_agent.capabilities.perf_tuning.tools import (
    analyze_frames,
    parse_systrace,
    trace_filter,
)
from embedded_device_agent.core.models import (
    ArtifactRef,
    KnowledgeRecord,
    RawTraceArtifact,
)
from embedded_device_agent.core.services import CoreServices

__all__ = ["build_tuning_graph", "DEFAULT_CAPTURE_DURATION_S"]

# 单次采集的默认时长（秒）；AppConfig 未显式承载采集时长，故在此给出可复现默认值。
DEFAULT_CAPTURE_DURATION_S = 2.0

# 由 ArtifactRef 重建 RawTraceArtifact 供确定性 trace_filter 使用时，只有 path 被真正
# 读取；其余字段以确定性占位值填充，避免引入非确定的时钟/环境依赖（Property 8 精神）。
_PLACEHOLDER_CAPTURED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _artifact_size(path: Path) -> int:
    """读取工件体量；尚未落盘（如 fake 占位路径）时记 0，不影响引用语义。"""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _run_id(state: TuningRunState) -> str:
    """据线程与迭代轮次派生本轮 run_id，供采集工件与本 run 关联（需求 2.2）。"""
    thread_id = state.get("thread_id") or "run"
    return f"{thread_id}-iter{state.get('iteration', 0)}"


def build_tuning_graph(
    services: CoreServices,
    *,
    collector: Collector | None = None,
    analyzer: Analyzer | None = None,
    instrumentor: Instrumentor | None = None,
    optimizer: Optimizer | None = None,
    capture_duration_s: float = DEFAULT_CAPTURE_DURATION_S,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """构建并编译调优闭环 subgraph（需求 1/6/8/9/10/11）。

    Parameters
    ----------
    services:
        共享底座；设备 / LLM / Retriever / Checkpointer / Store / Config 均由此注入，
        使闭环可用 fake 完整离线驱动（需求 18.4）。
    collector / analyzer / instrumentor / optimizer:
        可选注入的子智能体实例；缺省时由 ``services`` 构造。注入点便于测试以 fake
        子智能体驱动完整闭环（子智能体内部再以 FakeDeviceIO/FakeLLMProvider 替换副作用）。
    capture_duration_s:
        每轮采集时长（秒）。
    checkpointer:
        可选覆盖的 Checkpointer；缺省用 ``services.checkpointer`` 使运行可恢复
        （需求 10.1/10.2）。

    Returns
    -------
    CompiledStateGraph
        已接入 Checkpointer、状态模式为 :class:`TuningRunState` 的调优闭环图。
    """

    cfg = services.config
    collector = collector or Collector(services.device, cfg.device)
    analyzer = analyzer or Analyzer(services.llm)
    instrumentor = instrumentor or Instrumentor(services.llm, services.device)
    optimizer = optimizer or Optimizer(services.llm, services.device)
    retriever = services.retriever

    # ----------------------------------------------------------------- 节点
    def intake(state: TuningRunState) -> dict[str, Any]:
        """创建 Tuning_Run 并记录场景；缺场景则先请求（需求 1.1/1.2）。"""
        scenario = state.get("scenario") or ""
        if not scenario.strip():
            # 需求 1.2：缺少场景时先请求补充；状态由 Checkpointer 保留、可恢复。
            scenario = interrupt(
                {"reason": "missing_scenario", "prompt": "请提供要复现/调优的场景描述。"}
            )
        return {
            "scenario": scenario,
            "iteration": 0,
            "done": False,
            "instrumentation_history": state.get("instrumentation_history") or [],
            "optimizations_applied": state.get("optimizations_applied") or [],
            "max_iterations": state.get("max_iterations") or cfg.max_iterations,
            "mode": state.get("mode") or cfg.mode,
        }

    def recall(state: TuningRunState) -> dict[str, Any]:
        """按场景语义召回历史知识并纳入上下文（需求 1.3/9.3）。"""
        records = retriever.recall(state.get("scenario", ""), cfg.retriever.top_k)
        return {"recalled_knowledge": records}

    def plan(state: TuningRunState) -> dict[str, Any]:
        """规划本轮采集（骨架直通；具体策略由后续 LLM 编排细化）。"""
        return {}

    def collect(state: TuningRunState) -> dict[str, Any]:
        """指挥 Collector 采集本场景 trace，写入原始日志工件引用（需求 2）。"""
        result = collector.collect(
            run_id=_run_id(state),
            duration_s=capture_duration_s,
        )
        return {"latest_artifact": result.artifact}

    def filter_parse(state: TuningRunState) -> dict[str, Any]:
        """清洗原始日志为 systrace 工件（需求 3）；解析在 analyze 节点进行。"""
        ref = state.get("latest_artifact")
        if ref is None:
            raise ValueError("filter_parse 缺少 latest_artifact（应先 collect）。")
        raw = RawTraceArtifact(
            run_id=ref.run_id,
            path=ref.path,
            captured_at=_PLACEHOLDER_CAPTURED_AT,
            duration_s=0.0,
            baud=cfg.device.baud,
        )
        filtered = trace_filter(raw)
        clean = ArtifactRef(
            run_id=ref.run_id,
            kind="systrace",
            path=Path(filtered.clean_path),
            size_bytes=_artifact_size(Path(filtered.clean_path)),
        )
        return {"latest_artifact": clean}

    def analyze(state: TuningRunState) -> dict[str, Any]:
        """解析调用树、分析慢帧/热点并由 Analyzer 概括瓶颈（需求 4/5）。"""
        ref = state.get("latest_artifact")
        if ref is None:
            raise ValueError("analyze 缺少 latest_artifact（应先 filter_parse）。")
        tree = parse_systrace(ref)
        report = analyze_frames(tree, cfg, source=ref, scenario=state.get("scenario", ""))
        report = analyzer.analyze(report)
        updates: dict[str, Any] = {"latest": report}
        if state.get("baseline") is None:
            updates["baseline"] = report
        return updates

    def decide(state: TuningRunState) -> dict[str, Any]:
        """每轮 ``iteration++`` 并对照 ``max_iterations``（需求 6.5/11.4/11.5）。"""
        return {"iteration": state.get("iteration", 0) + 1}

    def instrument(state: TuningRunState) -> dict[str, Any]:
        """下钻埋点（骨架）：评估精度并记录一次埋点轮次，随后回到 collect 复测。"""
        precision = instrumentor.report_precision()
        history = list(state.get("instrumentation_history") or [])
        history.append(f"iter{state.get('iteration', 0)}:{precision.message}")
        return {"instrumentation_history": history}

    def optimize(state: TuningRunState) -> dict[str, Any]:
        """针对叶子热点提出/应用优化（需求 7）；应用后回到 collect 验证复测。"""
        report = state.get("latest")
        hotspot = report.hotspots[0] if (report and report.hotspots) else None
        result = optimizer.run(hotspot, state.get("mode") or "full_auto", report=report)
        if result.status == "applied" and result.applied is not None:
            applied = list(state.get("optimizations_applied") or [])
            applied.append(result.applied.target_func)
            return {"optimizations_applied": applied}
        return {}

    def halt(state: TuningRunState) -> dict[str, Any]:
        """达到迭代上限：停止闭环，转入 memorize 报告最佳证据（需求 6.5/11.5）。"""
        return {}

    def memorize(state: TuningRunState) -> dict[str, Any]:
        """沉淀本次运行知识并置 done（需求 8.4/9.1）；无慢帧/达标或收尾均在此落地。"""
        report = state.get("latest")
        scenario = state.get("scenario", "")
        applied = state.get("optimizations_applied") or []
        if report is not None:
            top = report.hotspots[0].func if report.hotspots else ""
            record = KnowledgeRecord(
                symptom=(report.summary or scenario or "性能调优")[:200],
                root_cause=top or "未定位到明确叶子热点",
                optimization=", ".join(applied) or "无（未达标或无需优化）",
                effect=(
                    "未检测到慢帧"
                    if report.no_slow_frames
                    else f"P95 帧耗时 {report.p95_frame_us}us"
                ),
                scenario=scenario,
                created_at=datetime.now(timezone.utc),
            )
            retriever.persist(record)
        return {"done": True}

    # --------------------------------------------------------------- 路由
    def route_after_decide(state: TuningRunState) -> str:
        """decide 后分支：达标→memorize；超限→halt；有热点→optimize；否则→instrument。"""
        report = state.get("latest")
        if report is not None and report.no_slow_frames:
            return "memorize"
        if state.get("iteration", 0) >= (state.get("max_iterations") or cfg.max_iterations):
            return "halt"
        if report is not None and report.hotspots:
            return "optimize"
        return "instrument"

    # --------------------------------------------------------------- 装配
    graph: StateGraph = StateGraph(TuningRunState)
    graph.add_node("intake", intake)
    graph.add_node("recall", recall)
    graph.add_node("plan", plan)
    graph.add_node("collect", collect)
    graph.add_node("filter_parse", filter_parse)
    graph.add_node("analyze", analyze)
    graph.add_node("decide", decide)
    graph.add_node("instrument", instrument)
    graph.add_node("optimize", optimize)
    graph.add_node("halt", halt)
    graph.add_node("memorize", memorize)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "recall")
    graph.add_edge("recall", "plan")
    graph.add_edge("plan", "collect")
    graph.add_edge("collect", "filter_parse")
    graph.add_edge("filter_parse", "analyze")
    graph.add_edge("analyze", "decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "instrument": "instrument",
            "optimize": "optimize",
            "memorize": "memorize",
            "halt": "halt",
        },
    )
    # instrument / optimize 之后回到 collect 复测，形成闭环（design.md）。
    graph.add_edge("instrument", "collect")
    graph.add_edge("optimize", "collect")
    graph.add_edge("halt", "memorize")
    graph.add_edge("memorize", END)

    return graph.compile(checkpointer=checkpointer or services.checkpointer)
