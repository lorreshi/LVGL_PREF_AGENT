"""共享 Pydantic v2 数据契约。

跨层传递的数据契约集中定义于此，保证：

* 可独立构造、可断言（支撑三级测试策略）；
* 可无损 JSON 序列化（支撑语言无关 API 边界，Property 9）；
* 大日志留在磁盘，state / 工具间只传 ``ArtifactRef`` 引用而非内容
  （Property 12，见 design.md "大日志处理与上下文预算"）。

字段严格对照 design.md 的 "Data Models" 章节，含微秒精度时间戳
（``ts_us`` / ``*_us``）与 ``ArtifactRef`` 引用语义。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "ArtifactRef",
    "RawTraceArtifact",
    "SystraceEvent",
    "FilterResult",
    "CallTreeNode",
    "ParseResult",
    "HotspotEntry",
    "SlowFrame",
    "FrameReport",
    "TraceQuery",
    "TraceSlice",
    "KnowledgeRecord",
]


# ---- Artifact 引用（大日志留在磁盘，state/工具间只传引用，不传内容）----
class ArtifactRef(BaseModel):
    """指向落盘工件的轻量引用。

    工具之间、以及运行状态中传递的都是该引用（run_id + kind + path），
    而非日志内容本身，从而使 Checkpointer 持久化的状态保持小体量
    （Property 12：原始日志不进上下文）。
    """

    run_id: str
    kind: Literal["raw_log", "systrace", "call_tree"]
    path: Path
    size_bytes: int


# ---- 采集 / trace 原始数据 ----
class RawTraceArtifact(BaseModel):
    """一次串口采集产出的原始 LVGL profiler 日志工件。"""

    run_id: str
    path: Path
    captured_at: datetime
    duration_s: float
    baud: int


class SystraceEvent(BaseModel):
    """单条 systrace 事件：``B|tid|func`` 或 ``E|tid|func``，附微秒时间戳。"""

    kind: Literal["B", "E"]  # begin / end
    tid: int
    func: str
    ts_us: int  # 微秒时间戳，保留亚毫秒精度（需求 4.5）


class FilterResult(BaseModel):
    """Trace_Filter 输出：清洗后 systrace 路径与保留/排除计数。"""

    clean_path: Path
    retained: int
    excluded: int
    # 受损区间列表（起止行/偏移），交错破坏 B/E 配对处（需求 3.2/3.3）
    corrupted_regions: list[tuple[int, int]] = Field(default_factory=list)


# ---- 解析后的调用树 ----
class CallTreeNode(BaseModel):
    """按线程组织的调用树节点，耗时以微秒表示。"""

    func: str
    tid: int
    begin_us: int
    end_us: int
    duration_us: int
    children: list["CallTreeNode"] = Field(default_factory=list)


class ParseResult(BaseModel):
    """Trace_Parser 输出：按 tid 分组的调用树 + 未匹配事件。"""

    trees_by_tid: dict[int, list[CallTreeNode]] = Field(default_factory=dict)
    unmatched: list[SystraceEvent] = Field(default_factory=list)  # 需求 4.3


# ---- 分析结果 ----
class HotspotEntry(BaseModel):
    """热点函数聚合统计（火焰图字段：函数名 + 聚合耗时）。"""

    func: str
    total_us: int
    call_count: int
    rank: int


class SlowFrame(BaseModel):
    """一帧超预算的慢帧记录。"""

    index: int
    duration_us: int
    dominant_func: str


class FrameReport(BaseModel):
    """Frame_Analyzer 输出的小体量摘要——唯一进入 LLM context 的分析产物。

    仅含超预算慢帧、top-N 热点与聚合统计（Property 12）；序列化保留渲染
    火焰图所需字段：函数名、时间戳/耗时、线程标识（Property 9，需求 15.4）。
    """

    scenario: str
    target_fps: int
    frame_budget_us: int
    slow_frames: list[SlowFrame] = Field(default_factory=list)  # 仅超预算帧
    hotspots: list[HotspotEntry] = Field(default_factory=list)  # 仅 top-N
    total_frames: int  # 聚合统计（不逐帧进 context）
    p95_frame_us: int
    source: ArtifactRef  # 指向调用树 artifact，供 query_trace 下钻
    summary: str | None = None  # Analyzer 概括，需求 5.3
    no_slow_frames: bool = False  # 需求 5.4


# ---- 按需下钻检索（大日志上下文预算机制 3）----
class TraceQuery(BaseModel):
    """按坐标（帧号/函数/线程/时间窗）下钻某一小段调用树的查询。"""

    frame_index: int | None = None
    func: str | None = None
    tid: int | None = None
    time_window_us: tuple[int, int] | None = None
    max_nodes: int = 200  # 单次返回上限，控制 context 增长


class TraceSlice(BaseModel):
    """query_trace 返回的调用树切片，``truncated`` 标记是否达到上限被截断。"""

    nodes: list[CallTreeNode] = Field(default_factory=list)
    truncated: bool = False


# ---- 记忆 ----
class KnowledgeRecord(BaseModel):
    """一条长期知识：症状 → 根因 → 优化 → 实测效果（需求 8.4/9.1）。"""

    id: str | None = None
    symptom: str
    root_cause: str
    optimization: str
    effect: str  # 基线 vs 优化后之差（需求 8.4/9.1）
    scenario: str
    created_at: datetime
