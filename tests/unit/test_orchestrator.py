"""任务 12.2：Orchestrator 工具装配单元测试 + Property 8。

对照 design.md「编排层与智能体层 → Orchestrator」与需求 13，在
``build_orchestrator_tools`` 层面自省并驱动装配到主体上的三类工具，全程离线
（注入 ``FakeDeviceIO`` / ``FakeLLMProvider`` / ``InMemoryStore``），不构建需要
tool-calling 模型的完整 ReAct 主体（FakeListChatModel 不实现 bind_tools）。

单元测试
* **子智能体 vs 普通工具区分**（需求 13.1 / 13.2）：装配出的工具集按名称/数量
  干净地划分为「子智能体工具 / 确定性普通工具 / 记忆工具」三类。
* **返回结果写入 state**（需求 13.3）：调用确定性工具 ``trace_filter_tool``（state
  持有指向录制原始日志的 raw_log ArtifactRef）返回一个 ``Command``，其 ``update``
  把清洗结果写入 ``latest_artifact``（变为 systrace）并追加一条 ``ToolMessage``。
* **错误处理路径**（需求 13.4）：对缺少 ``latest_artifact`` 的 state 调用工具触发
  异常，断言 ``tool_errors`` 被填充，且 full_auto 与 half_auto 两种模式下续行指引
  不同。

属性测试（hypothesis）
* **Property 8：确定性核心无副作用** —— 作为普通工具挂载的确定性工具，在相同输入
  state 上多次调用产出完全一致的结果（同一 systrace 工件引用），且不改动传入的
  state（无副作用）。
  **Validates: Requirements 13.1, 13.2, 13.3, 13.4**
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from embedded_device_agent.capabilities.perf_tuning.orchestrator import (
    build_orchestrator_tools,
)
from embedded_device_agent.core.config.models import (
    AppConfig,
    DeviceConfig,
    LLMConfig,
    RetrieverConfig,
)
from embedded_device_agent.core.memory.backends.local_store import LocalStoreRetriever
from embedded_device_agent.core.models import ArtifactRef
from embedded_device_agent.core.services import CoreServices

from tests.fixtures import RECORDED_CAPTURE_LOG
from tests.harness.fakes import FakeDeviceIO, FakeLLMProvider

# 装配到主体上的工具按设计分为三类（名称即工具函数名）。
_SUBAGENT_TOOLS = {"collector", "analyzer", "instrumentor", "optimizer"}
_DETERMINISTIC_TOOLS = {
    "trace_filter_tool",
    "trace_parser_tool",
    "frame_analyzer_tool",
    "query_trace_tool",
}
_MEMORY_TOOLS = {"recall_knowledge", "persist_knowledge"}


# --------------------------------------------------------------------------- #
# 辅助：构造离线 CoreServices 与工具集自省
# --------------------------------------------------------------------------- #
def _app_config() -> AppConfig:
    """构造一份最小但合法的 AppConfig（device/llm 由 fake 替换，字段仅取占位值）。"""
    return AppConfig(
        llm=LLMConfig(type="fake", model="fake-model", api_key_env="X"),
        device=DeviceConfig(type="fake", baud=115200, max_safe_baud=921600),
        retriever=RetrieverConfig(type="local_store", top_k=5),
        frame_budget_us=16667,
        mode="full_auto",
        max_iterations=5,
    )


def _services() -> CoreServices:
    """直接注入 fake 底座装配 CoreServices（不经工厂，避免真实硬件/网络/LLM）。"""
    config = _app_config()
    store = InMemoryStore()
    return CoreServices(
        device=FakeDeviceIO(),
        llm=FakeLLMProvider(responses=["ok"]),
        retriever=LocalStoreRetriever(config.retriever, store=store),
        checkpointer=InMemorySaver(),
        store=store,
        config=config,
    )


def _tools_by_name(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {t.name: t for t in tools}


def _raw_log_state(raw_path: Path, *, mode: str = "full_auto") -> dict:
    """构造一个持有 raw_log ArtifactRef 的运行状态。"""
    ref = ArtifactRef(
        run_id="t-iter0",
        kind="raw_log",
        path=raw_path,
        size_bytes=raw_path.stat().st_size,
    )
    return {
        "thread_id": "t",
        "scenario": "scroll",
        "mode": mode,
        "iteration": 0,
        "max_iterations": 5,
        "latest_artifact": ref,
    }


def _invoke(tool: BaseTool, state: dict, *, tool_call_id: str = "call-1", **kwargs):
    """按 LangChain ToolCall 形式调用工具：注入 state 与 tool_call_id。"""
    return tool.invoke(
        {
            "type": "tool_call",
            "name": tool.name,
            "id": tool_call_id,
            "args": {"state": state, **kwargs},
        }
    )


def _copy_raw_log(dest_dir: Path) -> Path:
    """把录制原始日志复制到隔离目录，避免 trace_filter 写出物污染 fixtures。"""
    dest = dest_dir / "raw.log"
    shutil.copyfile(RECORDED_CAPTURE_LOG, dest)
    return dest


# --------------------------------------------------------------------------- #
# 1) 子智能体 vs 普通工具 vs 记忆工具的划分（需求 13.1 / 13.2）
# --------------------------------------------------------------------------- #
def test_tool_set_partitions_by_category() -> None:
    """装配出的 10 个工具按名称干净地划分为三类（子智能体/确定性/记忆）。"""
    tools = build_orchestrator_tools(_services())
    names = {t.name for t in tools}

    # 恰好 10 个工具，且每类齐全、彼此不重叠。
    assert len(tools) == 10
    assert names == _SUBAGENT_TOOLS | _DETERMINISTIC_TOOLS | _MEMORY_TOOLS
    assert _SUBAGENT_TOOLS <= names  # 子智能体工具（需求 13.1）
    assert _DETERMINISTIC_TOOLS <= names  # 确定性普通工具（需求 13.2）
    assert _MEMORY_TOOLS <= names  # 记忆工具
    # 三类互不相交。
    assert not (_SUBAGENT_TOOLS & _DETERMINISTIC_TOOLS)
    assert not (_SUBAGENT_TOOLS & _MEMORY_TOOLS)
    assert not (_DETERMINISTIC_TOOLS & _MEMORY_TOOLS)


def test_all_assembled_tools_are_base_tools() -> None:
    """无论哪一类，装配产物都是可挂载到主体的 ``BaseTool``。"""
    tools = build_orchestrator_tools(_services())
    assert all(isinstance(t, BaseTool) for t in tools)


# --------------------------------------------------------------------------- #
# 2) 工具返回结果写入 state 再决策（需求 13.3）
# --------------------------------------------------------------------------- #
def test_deterministic_tool_writes_result_into_state(tmp_path: Path) -> None:
    """trace_filter_tool 把清洗结果写回 latest_artifact（systrace）并附 ToolMessage。"""
    tools = _tools_by_name(build_orchestrator_tools(_services()))
    raw_path = _copy_raw_log(tmp_path)
    state = _raw_log_state(raw_path)

    command = _invoke(tools["trace_filter_tool"], state, tool_call_id="c-filter")

    assert isinstance(command, Command)
    update = command.update
    # 结果写入状态：latest_artifact 从 raw_log 变为已清洗的 systrace 工件引用。
    new_artifact = update["latest_artifact"]
    assert isinstance(new_artifact, ArtifactRef)
    assert new_artifact.kind == "systrace"
    assert new_artifact.run_id == "t-iter0"
    assert new_artifact.path.exists()
    # 附一条 ToolMessage 供主体下一步决策读取（需求 13.3）。
    messages = update["messages"]
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "c-filter"
    # 录制日志 B/E 完整配对：14 条事件全部保留、无排除。
    assert "保留 14" in msg.content
    assert "排除 0" in msg.content


def test_deterministic_tool_result_flows_to_next_tool(tmp_path: Path) -> None:
    """写回的 systrace 工件可被下一确定性工具消费（先落状态、再决策的闭环）。"""
    tools = _tools_by_name(build_orchestrator_tools(_services()))
    raw_path = _copy_raw_log(tmp_path)
    state = _raw_log_state(raw_path)

    filtered = _invoke(tools["trace_filter_tool"], state)
    # 用写回的 systrace 更新状态，再驱动 trace_parser_tool。
    state["latest_artifact"] = filtered.update["latest_artifact"]

    parsed = _invoke(tools["trace_parser_tool"], state, tool_call_id="c-parse")
    assert isinstance(parsed, Command)
    parse_msg = parsed.update["messages"][0]
    assert isinstance(parse_msg, ToolMessage)
    assert "解析完成" in parse_msg.content
    # 解析产物不写入 state（Property 12）：仅返回消息，无 latest_artifact 覆写。
    assert "latest_artifact" not in parsed.update


# --------------------------------------------------------------------------- #
# 3) 工具报错按模式决定下一步（需求 13.4）
# --------------------------------------------------------------------------- #
def test_error_path_records_error_and_differs_by_mode() -> None:
    """缺 latest_artifact 触发异常：tool_errors 被填充，且续行指引按模式不同。"""
    tools = _tools_by_name(build_orchestrator_tools(_services()))

    full_auto_state = {"mode": "full_auto"}  # 无 latest_artifact → 触发 ValueError
    half_auto_state = {"mode": "half_auto"}

    full_cmd = _invoke(tools["trace_filter_tool"], full_auto_state, tool_call_id="e1")
    half_cmd = _invoke(tools["trace_filter_tool"], half_auto_state, tool_call_id="e2")

    # 两种模式都把错误记入 tool_errors（需求 13.4），且不覆写 latest_artifact。
    for cmd in (full_cmd, half_cmd):
        assert isinstance(cmd, Command)
        assert len(cmd.update["tool_errors"]) == 1
        assert "trace_filter 执行失败" in cmd.update["tool_errors"][0]
        assert "latest_artifact" not in cmd.update

    full_msg = full_cmd.update["messages"][0].content
    half_msg = half_cmd.update["messages"][0].content
    # full_auto：提示自主决定；half_auto：提示暂停交人工介入 —— 指引确实不同。
    assert "full_auto" in full_msg
    assert "自主决定" in full_msg
    assert "half_auto" in half_msg
    assert "人工介入" in half_msg
    assert full_msg != half_msg


def test_error_path_appends_to_existing_tool_errors() -> None:
    """已有错误时，新错误被追加而非覆盖（错误累计，需求 13.4）。"""
    tools = _tools_by_name(build_orchestrator_tools(_services()))
    state = {"mode": "full_auto", "tool_errors": ["先前错误"]}

    cmd = _invoke(tools["trace_filter_tool"], state, tool_call_id="e3")

    errors = cmd.update["tool_errors"]
    assert errors[0] == "先前错误"
    assert len(errors) == 2
    # 未就地改动传入 state 的错误列表（返回的是新列表）。
    assert state["tool_errors"] == ["先前错误"]


# --------------------------------------------------------------------------- #
# 4) Property 8：确定性核心无副作用（确定性工具作为普通工具挂载）
# --------------------------------------------------------------------------- #
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    tool_call_id=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12
    ),
    mode=st.sampled_from(["full_auto", "half_auto"]),
)
def test_property8_deterministic_tool_no_side_effects(
    tmp_path_factory, tool_call_id: str, mode: str
) -> None:
    """Property 8：确定性工具多次调用产出一致结果，且不改动传入 state。

    **Validates: Requirements 13.1, 13.2, 13.3, 13.4**

    ``trace_filter_tool`` 作为普通工具挂载（非 LLM 子智能体）：给定相同输入 state，
    两次调用返回**同一** systrace 工件引用（run_id / kind / 清洗计数一致），且调用
    前后传入 state 不被就地修改（无副作用）。tool_call_id 与 mode 属确定性无关维度，
    变化它们不应改变结果。
    """
    tmp_path = tmp_path_factory.mktemp("prop8")
    tools = _tools_by_name(build_orchestrator_tools(_services()))
    raw_path = _copy_raw_log(tmp_path)
    state = _raw_log_state(raw_path, mode=mode)
    snapshot = copy.deepcopy(state)

    cmd1 = _invoke(tools["trace_filter_tool"], state, tool_call_id=tool_call_id)
    cmd2 = _invoke(tools["trace_filter_tool"], state, tool_call_id=tool_call_id)

    a1 = cmd1.update["latest_artifact"]
    a2 = cmd2.update["latest_artifact"]
    # 确定性：两次结果的工件引用完全一致（同 run_id / kind / path）。
    assert a1 == a2
    assert a1.kind == "systrace"
    # 附带的 ToolMessage 正文（含保留/排除计数）也逐字一致。
    assert cmd1.update["messages"][0].content == cmd2.update["messages"][0].content
    # 无副作用：传入 state 未被就地修改（仍是原始 raw_log 引用与字段）。
    assert state == snapshot
    assert state["latest_artifact"].kind == "raw_log"
