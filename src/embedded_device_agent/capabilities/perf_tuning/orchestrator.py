"""Capability A 的 Orchestrator 工具装配（任务 12.1）。

严格对照 design.md「编排层与智能体层 → Orchestrator」与需求 13：

主智能体（Orchestrator）用 LangGraph 预置的 **ReAct 主体**构建。本项目安装的
``langgraph==1.2.8`` 提供的官方「subagents-as-tools」入口是
:func:`langgraph.prebuilt.create_react_agent`——本仓库**未安装** ``langchain`` 包，
因此**没有** ``langchain.agents.create_agent`` 符号；此处使用实际可用的
``create_react_agent``（其语义即设计中所述的 ``create_agent`` 主体 + 工具装配）。

装配到同一主体上的三类工具：

* **子智能体工具**（subagents-as-tools，需求 13.1）：``collector`` / ``analyzer`` /
  ``instrumentor`` / ``optimizer``——每个都是对相应 LLM 子智能体一次「委派」的封装，
  推理/判断留在子智能体内部完成。
* **确定性普通工具**（需求 13.2）：``trace_filter`` / ``trace_parser`` /
  ``frame_analyzer`` / ``query_trace``——纯输入输出的确定性核心，不作为 LLM 子智能体
  （Property 8）。
* **记忆工具**：``recall_knowledge`` / ``persist_knowledge``——经注入的 Retriever
  完成长期记忆的语义召回与持久化。

关键行为：

* **工具返回后写入 state 再决策**（需求 13.3）：每个工具都返回一个
  :class:`~langgraph.types.Command`，把结构化结果写入 :class:`TuningRunState` 的对应
  字段（``latest_artifact`` / ``baseline`` / ``latest`` / ``recalled_knowledge`` /
  ``instrumentation_history`` / ``optimizations_applied``），随附一条 ``ToolMessage``
  供主体在下一步决策时读取；Orchestrator 因此**先落状态、再决策**。
* **工具报错按模式决定下一步**（需求 13.4）：任一工具体内异常都被捕获并记入
  ``tool_errors``，再依 ``TuningRunState.mode`` 给出不同的续行指引——``full_auto``
  提示主体自主决定（重试 / 换路径 / 收尾），``half_auto`` 提示暂停并交人工介入。

一切副作用皆经注入的 :class:`CoreServices` 完成（设备、LLM、Retriever、
Checkpointer、Store 均来自共享底座）；因此本装配在测试中可用
``FakeDeviceIO`` / ``FakeLLMProvider`` / ``InMemoryStore`` 完整替换而离线自测。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict  # noqa: F401 (TypedDict re-exported)

from embedded_device_agent.capabilities.perf_tuning.agents import (
    Analyzer,
    Collector,
    Instrumentor,
    Optimizer,
)
from embedded_device_agent.capabilities.perf_tuning.agents.instrumentor import (
    InstrumentationPoint,
)
from embedded_device_agent.capabilities.perf_tuning.agents.optimizer import SourceEditor
from embedded_device_agent.capabilities.perf_tuning.state import TuningRunState
from embedded_device_agent.capabilities.perf_tuning.tools import (
    analyze_frames,
    parse_systrace,
    query_trace,
    trace_filter,
)
from embedded_device_agent.core.device.models import InputEvent
from embedded_device_agent.core.models import (
    ArtifactRef,
    FrameReport,
    KnowledgeRecord,
    RawTraceArtifact,
    TraceQuery,
)
from embedded_device_agent.core.services import CoreServices

__all__ = ["OrchestratorState", "build_orchestrator", "ORCHESTRATOR_SYSTEM_PROMPT"]


# 采集/清洗工件在 state 中只以 ``ArtifactRef`` 引用形式流转（Property 12）；重建
# ``RawTraceArtifact`` 供确定性 ``trace_filter`` 使用时，只有 ``path`` 被真正读取，
# 其余字段用确定性占位值填充，避免引入非确定的时钟/环境依赖。
_PLACEHOLDER_CAPTURED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


class OrchestratorState(TuningRunState, total=False):
    """Orchestrator（ReAct 主体）运行状态。

    在调优闭环状态 :class:`TuningRunState` 之上，追加 ReAct 主体所需的
    ``messages`` 通道（由 ``add_messages`` 归并）与 ``remaining_steps``（步数守卫），
    并以 ``tool_errors`` 累计工具错误（需求 13.4）。以 ``total=False`` 声明新增键为
    可选，便于上层 graph（任务 12.3）按阶段增量填充状态。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]
    tool_errors: list[str]


ORCHESTRATOR_SYSTEM_PROMPT = (
    "你是 LVGL 嵌入式性能调优的主编排智能体（Orchestrator）。你通过调用工具来推进"
    "一个调优闭环：采集(collector) → 清洗(trace_filter) → 解析(trace_parser) → "
    "分析(frame_analyzer) → 解读(analyzer) → 按需下钻(query_trace) → "
    "埋点细化(instrumentor) 或 源码优化(optimizer) → 复测验证 → 沉淀知识"
    "(persist_knowledge)。开始新问题时可先用 recall_knowledge 召回历史经验。\n\n"
    "工具分三类：\n"
    "1) 子智能体工具（collector/analyzer/instrumentor/optimizer）——把推理/判断委派"
    "给专职子智能体；\n"
    "2) 确定性普通工具（trace_filter/trace_parser/frame_analyzer/query_trace）——"
    "纯粹的数据处理，结果可复现；\n"
    "3) 记忆工具（recall_knowledge/persist_knowledge）——长期知识的召回与沉淀。\n\n"
    "重要原则：原始日志绝不进入你的上下文；你只看到工件引用与小体量的帧报告摘要，"
    "需要细节时用 query_trace 按坐标（帧号/函数/时间窗）按需翻页。\n"
    "每次工具返回后其结果已写入运行状态，请据最新状态决定下一步。\n"
    "当前自动化模式为「{mode}」：full_auto 下你可自主应用优化并复测；half_auto 下涉及"
    "烧录/应用优化的动作需要人工批准，遇到需要人工介入的情形请暂停并清晰汇报。"
)


# --------------------------------------------------------------------------- #
# 内部辅助
# --------------------------------------------------------------------------- #
def _mode_of(state: dict[str, Any]) -> str:
    """从运行状态读取自动化模式，缺省视为 ``full_auto``（需求 13.4）。"""
    return state.get("mode") or "full_auto"


def _run_id_of(state: dict[str, Any]) -> str:
    """据线程与迭代轮次派生本轮 run_id，供采集工件与本 run 关联（需求 2.2）。"""
    thread_id = state.get("thread_id") or "run"
    iteration = state.get("iteration", 0)
    return f"{thread_id}-iter{iteration}"


def _artifact_size(path: Path) -> int:
    """读取工件体量；尚未落盘（如 fake 占位路径）时记 0，不影响引用语义。"""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _ok(
    tool_call_id: str, message: str, updates: dict[str, Any] | None = None
) -> Command:
    """构造「成功」的状态增量：写入结果字段并附一条 ToolMessage（需求 13.3）。"""
    update: dict[str, Any] = dict(updates or {})
    update["messages"] = [ToolMessage(content=message, tool_call_id=tool_call_id)]
    return Command(update=update)


def _fail(
    tool_call_id: str,
    tool_name: str,
    exc: Exception,
    state: dict[str, Any],
) -> Command:
    """按自动化模式处理工具错误：记入 ``tool_errors`` 并给出续行指引（需求 13.4）。

    * ``full_auto``：提示主体据错误自主决定下一步（重试 / 换路径 / 收尾）。
    * ``half_auto``：提示暂停并交由人工介入。
    """
    mode = _mode_of(state)
    detail = f"{tool_name} 执行失败：{type(exc).__name__}: {exc}"
    if mode == "half_auto":
        guidance = "（half_auto 模式）请暂停闭环并将该错误交由开发者人工介入处理。"
    else:
        guidance = (
            "（full_auto 模式）请据此错误自主决定下一步：重试、改走其他路径或收尾报告。"
        )
    errors = list(state.get("tool_errors") or [])
    errors.append(detail)
    return Command(
        update={
            "tool_errors": errors,
            "messages": [
                ToolMessage(content=f"{detail}\n{guidance}", tool_call_id=tool_call_id)
            ],
        }
    )


def _as_artifact(value: Any) -> ArtifactRef:
    """把状态中的工件引用（可能已被序列化为 dict）归一为 :class:`ArtifactRef`。"""
    if isinstance(value, ArtifactRef):
        return value
    return ArtifactRef.model_validate(value)


def _require_artifact(state: dict[str, Any], *, kinds: tuple[str, ...]) -> ArtifactRef:
    """从状态取出 ``latest_artifact`` 并校验其 ``kind`` 属于期望集合。"""
    raw = state.get("latest_artifact")
    if raw is None:
        raise ValueError("运行状态中没有可用的工件（latest_artifact 为空）。")
    artifact = _as_artifact(raw)
    if artifact.kind not in kinds:
        raise ValueError(
            f"期望工件类型 {kinds}，但 latest_artifact 为 {artifact.kind!r}。"
        )
    return artifact


def _coerce_report(value: Any) -> FrameReport | None:
    """把状态中的 FrameReport（可能已被序列化为 dict）归一为模型实例。"""
    if value is None:
        return None
    if isinstance(value, FrameReport):
        return value
    return FrameReport.model_validate(value)


# --------------------------------------------------------------------------- #
# 子智能体工具（subagents-as-tools，需求 13.1）
# --------------------------------------------------------------------------- #
def _build_subagent_tools(
    services: CoreServices,
    *,
    source_editor: SourceEditor | None,
    tick_resolution_us: int,
) -> list[BaseTool]:
    """构造 4 个子智能体工具，每个封装对相应 LLM 子智能体的一次委派（需求 13.1）。

    子智能体实例在此**先行构造**并被下面的闭包直接捕获——推理/判断留在子智能体内部
    完成，工具层只负责把状态喂进去、把结果写回状态。
    """

    collector_agent = Collector(services.device, services.config.device)
    analyzer_agent = Analyzer(services.llm)
    instrumentor_agent = Instrumentor(
        services.llm, services.device, tick_resolution_us=tick_resolution_us
    )
    optimizer_agent = Optimizer(
        services.llm, services.device, source_editor=source_editor
    )

    @tool
    def collector(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        duration_s: float = 2.0,
        trigger_kind: str | None = None,
        trigger_params: dict[str, Any] | None = None,
    ) -> Command:
        """委派 Collector 子智能体采集本场景的 LVGL profiler trace（需求 2）。

        经注入的 DeviceIO 打开串口、按需注入输入并采集；必要时降波特率重采。采集
        完成后把**原始日志工件引用**写入 ``latest_artifact``（仅引用、不含内容）。
        """
        try:
            trigger = (
                InputEvent(kind=trigger_kind, params=trigger_params or {})
                if trigger_kind
                else None
            )
            result = collector_agent.collect(
                run_id=_run_id_of(state),
                duration_s=duration_s,
                trigger_event=trigger,
            )
        except Exception as exc:  # noqa: BLE001 - 按模式统一处理（需求 13.4）
            return _fail(tool_call_id, "collector", exc, state)

        msg = (
            f"采集完成：工件 {result.artifact.kind}@{result.artifact.path}，"
            f"波特率 {result.baud}，尝试 {result.attempts} 次，"
            f"完整={result.complete}（保留 {result.retained}/排除 {result.excluded}）。"
        )
        return _ok(tool_call_id, msg, {"latest_artifact": result.artifact})

    @tool
    def analyzer(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """委派 Analyzer 子智能体解读最新帧报告，产出带证据的瓶颈概括（需求 5.3）。

        读取状态中的 ``latest`` 帧报告，交由 Analyzer 填写 ``summary`` 后写回。
        """
        try:
            report = _coerce_report(state.get("latest"))
            if report is None:
                raise ValueError("尚无帧报告可解读（请先运行 frame_analyzer）。")
            analyzed = analyzer_agent.analyze(report)
        except Exception as exc:  # noqa: BLE001
            return _fail(tool_call_id, "analyzer", exc, state)
        return _ok(tool_call_id, f"瓶颈概括：{analyzed.summary}", {"latest": analyzed})

    @tool
    def instrumentor(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        points: list[dict],
        sources: dict[str, str] | None = None,
        reflash: bool = True,
    ) -> Command:
        """委派 Instrumentor 子智能体在粗粒度热点内插入更细的 profiler 埋点（需求 6）。

        ``points`` 为待插入的成对埋点点位（file/tag/begin_line/end_line）；
        ``sources`` 可选地内联源码（file→源码文本），缺省时按 file 从磁盘读取。埋点后
        （``reflash``）经 DeviceIO 重编译并烧录，标签追加到 ``instrumentation_history``。
        """
        try:
            pts = [InstrumentationPoint.model_validate(p) for p in points]
            src = dict(sources) if sources else {}
            for p in pts:
                if p.file not in src:
                    src[p.file] = Path(p.file).read_text(encoding="utf-8")
            result = instrumentor_agent.instrument(src, pts, reflash=reflash)
        except Exception as exc:  # noqa: BLE001
            return _fail(tool_call_id, "instrumentor", exc, state)

        history = list(state.get("instrumentation_history") or [])
        history.extend(result.tags)
        msg = (
            f"已插入埋点标签 {result.tags}；精度评估：{result.precision.message}"
            f"（reflashed={result.reflashed}）。"
        )
        return _ok(tool_call_id, msg, {"instrumentation_history": history})

    @tool
    def optimizer(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        context: str = "",
    ) -> Command:
        """委派 Optimizer 子智能体针对已隔离叶子热点提出/应用源码优化（需求 7）。

        取状态中 ``latest`` 帧报告的首位热点作为叶子热点：full_auto 直接应用并重编译
        烧录（应用后目标函数追加到 ``optimizations_applied``）；half_auto 仅呈现待批准、
        不应用（需求 7.2 / 7.3）。
        """
        try:
            report = _coerce_report(state.get("latest"))
            hotspot = report.hotspots[0] if (report and report.hotspots) else None
            result = optimizer_agent.run(
                hotspot, _mode_of(state), report=report, context=context
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(tool_call_id, "optimizer", exc, state)

        if result.status == "applied" and result.applied is not None:
            applied = list(state.get("optimizations_applied") or [])
            applied.append(result.applied.target_func)
            return _ok(
                tool_call_id,
                f"已应用优化：{result.applied.summary}",
                {"optimizations_applied": applied},
            )
        if result.status == "pending_approval":
            return _ok(
                tool_call_id,
                "（half_auto）已生成优化提议待开发者批准，批准前不应用任何改动。",
            )
        return _ok(tool_call_id, "未发现可优化的叶子热点。")

    return [collector, analyzer, instrumentor, optimizer]


# --------------------------------------------------------------------------- #
# 确定性普通工具（需求 13.2）
# --------------------------------------------------------------------------- #
def _build_deterministic_tools(services: CoreServices) -> list[BaseTool]:
    """构造 4 个确定性普通工具：清洗 / 解析 / 帧分析 / 下钻检索（需求 13.2）。

    它们均为纯输入输出的确定性核心（Property 8），不作为 LLM 子智能体；结果写回状态
    后由 Orchestrator 决策。
    """

    cfg = services.config

    @tool
    def trace_filter_tool(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """把 ``latest_artifact``（原始日志）清洗为 systrace 工件（需求 3）。

        排除破坏 B/E 配对的交错事件，报告保留/排除计数与受损区间，并把已清洗
        systrace 工件引用写回 ``latest_artifact``。
        """
        try:
            raw_ref = _require_artifact(state, kinds=("raw_log",))
            raw = RawTraceArtifact(
                run_id=raw_ref.run_id,
                path=raw_ref.path,
                captured_at=_PLACEHOLDER_CAPTURED_AT,
                duration_s=0.0,
                baud=cfg.device.baud,
            )
            result = trace_filter(raw)
            clean = ArtifactRef(
                run_id=raw_ref.run_id,
                kind="systrace",
                path=Path(result.clean_path),
                size_bytes=_artifact_size(Path(result.clean_path)),
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(tool_call_id, "trace_filter", exc, state)

        msg = (
            f"清洗完成：systrace@{clean.path}，保留 {result.retained}/排除 "
            f"{result.excluded}，受损区间 {result.corrupted_regions}。"
        )
        return _ok(tool_call_id, msg, {"latest_artifact": clean})

    @tool
    def trace_parser_tool(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """把 ``latest_artifact``（systrace）解析为按线程调用树并报告未匹配事件（需求 4）。

        调用树本身不进 state（Property 12）；下游 ``frame_analyzer`` / ``query_trace``
        会从同一 systrace 工件确定性地重建。此处只报告未匹配事件计数与线程数。
        """
        try:
            ref = _require_artifact(state, kinds=("systrace", "call_tree"))
            parsed = parse_systrace(ref)
        except Exception as exc:  # noqa: BLE001
            return _fail(tool_call_id, "trace_parser", exc, state)

        msg = (
            f"解析完成：{len(parsed.trees_by_tid)} 个线程的调用树，"
            f"未匹配事件 {len(parsed.unmatched)} 条。"
        )
        return _ok(tool_call_id, msg)

    @tool
    def frame_analyzer_tool(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """把调用树分析为小体量慢帧/热点摘要并写入状态（需求 5.1 / 5.2 / 5.4）。

        从 ``latest_artifact``（systrace）解析后运行确定性 ``analyze_frames``，把
        ``FrameReport`` 写入 ``latest``；若尚无 ``baseline`` 则一并作为基线记录，供后续
        优化对比。
        """
        try:
            ref = _require_artifact(state, kinds=("systrace", "call_tree"))
            parsed = parse_systrace(ref)
            report = analyze_frames(
                parsed,
                cfg,
                source=ref,
                scenario=state.get("scenario", ""),
            )
        except Exception as exc:  # noqa: BLE001
            return _fail(tool_call_id, "frame_analyzer", exc, state)

        updates: dict[str, Any] = {"latest": report}
        if state.get("baseline") is None:
            updates["baseline"] = report
        msg = (
            f"帧分析完成：总帧 {report.total_frames}，慢帧 {len(report.slow_frames)}，"
            f"热点 {len(report.hotspots)}，P95 {report.p95_frame_us}us，"
            f"no_slow_frames={report.no_slow_frames}。"
        )
        return _ok(tool_call_id, msg, updates)

    @tool
    def query_trace_tool(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        frame_index: int | None = None,
        func: str | None = None,
        tid: int | None = None,
        time_window_us: tuple[int, int] | None = None,
        max_nodes: int = 200,
    ) -> Command:
        """按坐标从调用树 artifact 下钻取回一小段切片（需求 4.1 / 5.2）。

        这是「LLM 不读日志、只按需翻页」的落点：细节以 ToolMessage 返回供当次决策，
        不写入 state（保持状态小体量，Property 12）。
        """
        try:
            ref = _require_artifact(state, kinds=("systrace", "call_tree"))
            q = TraceQuery(
                frame_index=frame_index,
                func=func,
                tid=tid,
                time_window_us=time_window_us,
                max_nodes=max_nodes,
            )
            sliced = query_trace(ref, q)
        except Exception as exc:  # noqa: BLE001
            return _fail(tool_call_id, "query_trace", exc, state)

        preview = ", ".join(
            f"{n.func}({n.duration_us}us)" for n in sliced.nodes[:10]
        )
        msg = (
            f"下钻结果：命中 {len(sliced.nodes)} 个节点，truncated={sliced.truncated}。"
            f" 顶层节点：{preview or '(空)'}"
        )
        return _ok(tool_call_id, msg)

    return [trace_filter_tool, trace_parser_tool, frame_analyzer_tool, query_trace_tool]


# --------------------------------------------------------------------------- #
# 记忆工具
# --------------------------------------------------------------------------- #
def _build_memory_tools(services: CoreServices) -> list[BaseTool]:
    """构造记忆工具：语义召回 ``recall_knowledge`` 与知识沉淀 ``persist_knowledge``。"""

    retriever = services.retriever
    default_k = services.config.retriever.top_k

    @tool
    def recall_knowledge(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        symptom: str,
        k: int | None = None,
    ) -> Command:
        """按症状语义召回历史知识并写入 ``recalled_knowledge``（需求 1 / 9.2）。"""
        try:
            records = retriever.recall(symptom, k if k is not None else default_k)
        except Exception as exc:  # noqa: BLE001
            return _fail(tool_call_id, "recall_knowledge", exc, state)

        msg = f"召回 {len(records)} 条历史知识。" + "".join(
            f"\n- [{r.scenario}] {r.symptom} → {r.root_cause}" for r in records
        )
        return _ok(tool_call_id, msg, {"recalled_knowledge": records})

    @tool
    def persist_knowledge(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        symptom: str,
        root_cause: str,
        optimization: str,
        effect: str,
        scenario: str = "",
    ) -> Command:
        """把「症状→根因→优化→效果」沉淀为长期知识并返回其 id（需求 8.4 / 9.1）。"""
        try:
            record = KnowledgeRecord(
                symptom=symptom,
                root_cause=root_cause,
                optimization=optimization,
                effect=effect,
                scenario=scenario or state.get("scenario", ""),
                created_at=datetime.now(timezone.utc),
            )
            record_id = retriever.persist(record)
        except Exception as exc:  # noqa: BLE001
            return _fail(tool_call_id, "persist_knowledge", exc, state)

        return _ok(tool_call_id, f"已沉淀知识记录，id={record_id}。")

    return [recall_knowledge, persist_knowledge]


# --------------------------------------------------------------------------- #
# 主体装配
# --------------------------------------------------------------------------- #
def build_orchestrator_tools(
    services: CoreServices,
    *,
    source_editor: SourceEditor | None = None,
    tick_resolution_us: int = 1_000,
) -> list[BaseTool]:
    """装配 Orchestrator 的全部工具：子智能体 + 确定性 + 记忆（需求 13.1 / 13.2）。

    单独暴露以便测试自省工具集（区分子智能体工具与普通工具），也供上层 graph 复用。
    """
    return [
        *_build_subagent_tools(
            services,
            source_editor=source_editor,
            tick_resolution_us=tick_resolution_us,
        ),
        *_build_deterministic_tools(services),
        *_build_memory_tools(services),
    ]


def build_orchestrator(
    services: CoreServices,
    *,
    source_editor: SourceEditor | None = None,
    tick_resolution_us: int = 1_000,
    name: str = "perf_tuning_orchestrator",
) -> CompiledStateGraph:
    """用 ``create_react_agent`` 装配 Capability A 的 Orchestrator 主体（需求 13）。

    Parameters
    ----------
    services:
        共享底座 :class:`CoreServices`——设备 / LLM / Retriever / Checkpointer / Store
        / Config 均由此注入，使装配可用 fake 完整离线自测（需求 18.4）。
    source_editor:
        可选注入的源码编辑器，供 Optimizer 应用优化（测试可替换为 fake，避免触碰真实
        源码树）；缺省用 :class:`FileSourceEditor`。
    tick_resolution_us:
        目标平台默认时间戳/ tick 分辨率（微秒），供 Instrumentor 精度守卫（需求 6.6）。
    name:
        编译后主体名。

    Returns
    -------
    CompiledStateGraph
        已装配三类工具、注入系统提示与共享底座（Checkpointer/Store）的 ReAct 主体。
        状态模式为 :class:`OrchestratorState`（含调优闭环状态 + messages），每次工具
        返回后结果写入状态再决策（需求 13.3），工具报错按模式处理（需求 13.4）。
    """

    tools = build_orchestrator_tools(
        services,
        source_editor=source_editor,
        tick_resolution_us=tick_resolution_us,
    )

    def _prompt(state: OrchestratorState) -> list[AnyMessage]:
        """据当前自动化模式渲染系统提示，前置到消息序列（需求 11.1 / 13.4）。"""
        mode = state.get("mode") or "full_auto"
        system = SystemMessage(
            content=ORCHESTRATOR_SYSTEM_PROMPT.format(mode=mode)
        )
        return [system, *state.get("messages", [])]

    return create_react_agent(
        services.llm.get_chat_model(),
        tools,
        prompt=_prompt,
        state_schema=OrchestratorState,
        checkpointer=services.checkpointer,
        store=services.store,
        name=name,
    )
