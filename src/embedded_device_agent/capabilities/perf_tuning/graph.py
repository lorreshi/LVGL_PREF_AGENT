"""Capability A 调优闭环 subgraph（任务 12.3 + 12.4）。

严格对照 design.md「编排层与智能体层 → 调优闭环状态机」的节点与转移：

    [*] → intake → recall → plan → collect → filter_parse → analyze → decide
    decide → instrument   （需更细粒度）        → collect（重编译烧录后复测）
    decide → optimize      （已定位叶子热点）    → collect（应用优化后复测/验证）
    decide → memorize      （达标 / 无慢帧 / 验证为改进）→ [*]
    decide → halt          （达到 Max_Iteration_Limit）→ memorize → [*]

关键设计（对照需求 1/6/8/9/10/11）：

* **intake**（需求 1.1/1.2）：创建 Tuning_Run 并记录场景；**缺场景则先请求**
  （用 LangGraph ``interrupt`` 暂停，等待补充场景后恢复）。
* **recall**（需求 1.3/9.3）：按场景语义召回历史知识并纳入 ``recalled_knowledge``。
* **decide**（需求 6.5/8.2/8.3/11.4/11.5）：每轮 ``iteration++`` 并对照
  ``max_iterations``；对上一轮已应用的优化做**基线对比验证**——有改进则接受、
  无改进则标记无效并回退，在迭代上限内改选替代方案；``instrument`` / ``optimize``
  之后回到 ``collect`` 复测形成闭环；超限走 ``halt`` 报告最佳证据后收尾。
* **Half_Auto 人在环**（需求 10.3/11.3）：在配置的干预点（``collect`` 触发场景、
  ``instrument`` / ``optimize`` 烧录/批准）用 ``interrupt`` 暂停，状态由 Checkpointer
  保留，resume 后从断点继续。
* **知识沉淀**（需求 8.4/9.1）：验证为改进时在 ``memorize`` 持久化 Knowledge_Record。
* **可恢复**（需求 10.1/10.2）：图以注入的 Checkpointer 编译。

一切副作用皆经注入的 :class:`CoreServices` 完成；四个子智能体默认由 services 构造，
亦可显式注入以便测试用 fake 完整离线驱动闭环——节点直接调用子智能体与确定性工具，
**不依赖具备 tool-calling 能力的 LLM**。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    FrameReport,
    KnowledgeRecord,
    RawTraceArtifact,
)
from embedded_device_agent.core.services import CoreServices

__all__ = ["build_tuning_graph", "TuningLoopState", "DEFAULT_CAPTURE_DURATION_S"]

# 单次采集的默认时长（秒）；AppConfig 未显式承载采集时长，故在此给出可复现默认值。
DEFAULT_CAPTURE_DURATION_S = 2.0

# 由 ArtifactRef 重建 RawTraceArtifact 供确定性 trace_filter 使用时，只有 path 被真正
# 读取；其余字段以确定性占位值填充，避免引入非确定的时钟/环境依赖（Property 8 精神）。
_PLACEHOLDER_CAPTURED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


class TuningLoopState(TuningRunState, total=False):
    """闭环运行状态：在 :class:`TuningRunState` 之上追加验证/回退所需工作字段。

    以 ``total=False`` 声明新增键为可选，保持对外的 ``TuningRunState`` 契约不变：

    * ``pending_optimization``：上一轮已应用、待复测验证的优化目标函数；
    * ``pre_opt_p95_us``：应用该优化前的基线 P95 帧耗时，用于对比是否改进；
    * ``invalidated_optimizations``：验证为无改进而被回退/标记无效的优化（需求 8.3）；
    * ``last_verified_effective``：最近一次验证结论（改进/无改进），供收尾报告。
    """

    pending_optimization: str | None
    pre_opt_p95_us: int | None
    invalidated_optimizations: list[str]
    last_verified_effective: bool | None


def _artifact_size(path: Path) -> int:
    """读取工件体量；尚未落盘（如 fake 占位路径）时记 0，不影响引用语义。"""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _run_id(state: TuningLoopState) -> str:
    """据线程与迭代轮次派生本轮 run_id，供采集工件与本 run 关联（需求 2.2）。"""
    thread_id = state.get("thread_id") or "run"
    return f"{thread_id}-iter{state.get('iteration', 0)}"


def _is_improved(latest: FrameReport, pre_p95_us: int) -> bool:
    """判定复测是否较优化前有改进（需求 8.2）：无慢帧或 P95 帧耗时下降即为改进。"""
    if latest.no_slow_frames:
        return True
    return latest.p95_frame_us < pre_p95_us


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
        子智能体驱动完整闭环。
    capture_duration_s:
        每轮采集时长（秒）。
    checkpointer:
        可选覆盖的 Checkpointer；缺省用 ``services.checkpointer`` 使运行可恢复。

    Returns
    -------
    CompiledStateGraph
        已接入 Checkpointer、状态模式为 :class:`TuningLoopState` 的调优闭环图。
    """

    cfg = services.config
    collector = collector or Collector(services.device, cfg.device)
    analyzer = analyzer or Analyzer(services.llm)
    instrumentor = instrumentor or Instrumentor(services.llm, services.device)
    optimizer = optimizer or Optimizer(services.llm, services.device)
    retriever = services.retriever
    intervention_points = set(cfg.intervention_points or [])

    def _half_auto(state: TuningLoopState) -> bool:
        return (state.get("mode") or cfg.mode) == "half_auto"

    # ----------------------------------------------------------------- 节点
    def intake(state: TuningLoopState) -> dict[str, Any]:
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
            "invalidated_optimizations": state.get("invalidated_optimizations") or [],
            "pending_optimization": None,
            "pre_opt_p95_us": None,
            "last_verified_effective": None,
            "max_iterations": state.get("max_iterations") or cfg.max_iterations,
            "mode": state.get("mode") or cfg.mode,
        }

    def recall(state: TuningLoopState) -> dict[str, Any]:
        """按场景语义召回历史知识并纳入上下文（需求 1.3/9.3）。"""
        records = retriever.recall(state.get("scenario", ""), cfg.retriever.top_k)
        return {"recalled_knowledge": records}

    def plan(state: TuningLoopState) -> dict[str, Any]:
        """规划本轮采集（骨架直通；具体策略由后续 LLM 编排细化）。"""
        return {}

    def collect(state: TuningLoopState) -> dict[str, Any]:
        """指挥 Collector 采集本场景 trace，写入原始日志工件引用（需求 2）。

        Half_Auto 且配置了 ``collect`` 干预点时，先 ``interrupt`` 等待人工触发场景
        （需求 10.3/11.3），状态由 Checkpointer 保留、resume 后继续。
        """
        if _half_auto(state) and "collect" in intervention_points:
            interrupt(
                {
                    "reason": "manual_scenario_trigger",
                    "prompt": "（Half_Auto）请在设备上手动触发目标场景，然后恢复以开始采集。",
                    "scenario": state.get("scenario", ""),
                }
            )
        result = collector.collect(
            run_id=_run_id(state),
            duration_s=capture_duration_s,
        )
        return {"latest_artifact": result.artifact}

    def filter_parse(state: TuningLoopState) -> dict[str, Any]:
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

    def analyze(state: TuningLoopState) -> dict[str, Any]:
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

    def decide(state: TuningLoopState) -> dict[str, Any]:
        """每轮 ``iteration++``；对上一轮优化做基线对比验证（需求 6.5/8.2/8.3/11.4）。

        若上一轮已应用优化（``pending_optimization``），本轮复测后对比其应用前的 P95：
        有改进则接受并清除待验证标记（需求 8.2）；无改进则标记该优化无效、记入
        ``invalidated_optimizations`` 以便回退改选替代方案（需求 8.3）。
        """
        updates: dict[str, Any] = {"iteration": state.get("iteration", 0) + 1}

        pending = state.get("pending_optimization")
        latest = state.get("latest")
        if pending and latest is not None:
            pre_p95 = state.get("pre_opt_p95_us")
            improved = _is_improved(latest, pre_p95 if pre_p95 is not None else latest.p95_frame_us)
            updates["last_verified_effective"] = improved
            updates["pending_optimization"] = None
            updates["pre_opt_p95_us"] = None
            if not improved:
                # 需求 8.3：无改进 → 标记无效并回退，供后续改选替代方案。
                invalidated = list(state.get("invalidated_optimizations") or [])
                invalidated.append(pending)
                updates["invalidated_optimizations"] = invalidated
        return updates

    def instrument(state: TuningLoopState) -> dict[str, Any]:
        """下钻埋点（骨架）：评估精度并记录一次埋点轮次，随后回到 collect 复测。

        Half_Auto 且配置了 ``instrument`` 干预点时先 ``interrupt`` 等待人工烧录确认。
        """
        if _half_auto(state) and "instrument" in intervention_points:
            interrupt(
                {
                    "reason": "manual_flash_after_instrument",
                    "prompt": "（Half_Auto）已插入埋点，请人工重编译烧录后恢复以复测。",
                }
            )
        precision = instrumentor.report_precision()
        history = list(state.get("instrumentation_history") or [])
        history.append(f"iter{state.get('iteration', 0)}:{precision.message}")
        return {"instrumentation_history": history}

    def optimize(state: TuningLoopState) -> dict[str, Any]:
        """针对叶子热点提出/应用优化（需求 7）；记录待验证基线，回到 collect 复测。

        Half_Auto 且配置了 ``optimize`` 干预点时先 ``interrupt`` 等待人工批准
        （需求 7.3/11.3）。记录应用前的 P95 与目标函数，供 decide 复测后对比验证。
        """
        if _half_auto(state) and "optimize" in intervention_points:
            interrupt(
                {
                    "reason": "approve_optimization",
                    "prompt": "（Half_Auto）请审阅并批准优化提议后恢复以应用并复测。",
                }
            )
        report = state.get("latest")
        hotspot = report.hotspots[0] if (report and report.hotspots) else None
        result = optimizer.run(hotspot, state.get("mode") or "full_auto", report=report)
        updates: dict[str, Any] = {}
        if result.status == "applied" and result.applied is not None:
            applied = list(state.get("optimizations_applied") or [])
            applied.append(result.applied.target_func)
            updates["optimizations_applied"] = applied
            # 记录待验证：下一轮复测后由 decide 对比 P95 判定是否改进（需求 8.2）。
            updates["pending_optimization"] = result.applied.target_func
            updates["pre_opt_p95_us"] = report.p95_frame_us if report else None
        return updates

    def halt(state: TuningLoopState) -> dict[str, Any]:
        """达到迭代上限：停止闭环，转入 memorize 报告最佳证据（需求 6.5/11.5）。"""
        return {}

    def memorize(state: TuningLoopState) -> dict[str, Any]:
        """沉淀本次运行知识并置 done（需求 8.4/9.1）。

        仅在**验证为改进**（或无慢帧达标）时持久化一条 Knowledge_Record；无改进的收尾
        （halt 或优化无效）不写入误导性知识，只结束运行。
        """
        report = state.get("latest")
        scenario = state.get("scenario", "")
        applied = state.get("optimizations_applied") or []
        invalidated = set(state.get("invalidated_optimizations") or [])
        effective_opts = [f for f in applied if f not in invalidated]

        improved = bool(report is not None and report.no_slow_frames) or bool(
            state.get("last_verified_effective")
        )
        if report is not None and improved:
            top = report.hotspots[0].func if report.hotspots else ""
            record = KnowledgeRecord(
                symptom=(report.summary or scenario or "性能调优")[:200],
                root_cause=top or "未定位到明确叶子热点",
                optimization=", ".join(effective_opts) or "无需优化即达标",
                effect=(
                    "未检测到慢帧"
                    if report.no_slow_frames
                    else f"P95 帧耗时降至 {report.p95_frame_us}us"
                ),
                scenario=scenario,
                created_at=datetime.now(timezone.utc),
            )
            retriever.persist(record)
        return {"done": True}

    # --------------------------------------------------------------- 路由
    def route_after_decide(state: TuningLoopState) -> str:
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
    graph: StateGraph = StateGraph(TuningLoopState)
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
