"""Trace_Parser：已清洗 systrace 工件 → 按线程调用树（任务 8.3）。

作为**确定性普通工具**由 Orchestrator 调用（需求 13.2）。纯输入输出：给定相同的
已清洗 systrace 内容，恒产生相同的 ``ParseResult``，不产生任何外部副作用
（Property 8）。

输入为 ``Trace_Filter`` 产出的已清洗 systrace 工件（``FilterResult.clean_path``），
其事件行形如::

    LVGL-1 [0] 2892.002993: tracing_mark_write: B|1|lv_timer_handler
    LVGL-1 [0] 2892.003550: tracing_mark_write: E|1|lv_timer_handler

其中时间戳为「秒.微秒」（亚毫秒精度），``B`` / ``E`` 为函数进入 / 退出标记，
``tid`` 为线程号，``func`` 为函数名。

本工具行为（严格对照需求 4）：

* **需求 4.1**：把每个 ``B|tid|func`` / ``E|tid|func`` 事件解析为按线程（tid）组织
  的调用树（``ParseResult.trees_by_tid``）。
* **需求 4.2**：对一对匹配的 begin / end，函数耗时 = ``end_us - begin_us``
  （``CallTreeNode.duration_us``）。
* **需求 4.3**：某个 begin 事件在其所属线程内没有匹配的 end（收尾时仍未闭合），
  或某个 end 找不到匹配的 begin，则报告该未匹配事件（``ParseResult.unmatched``）
  并将其排除出调用树。
* **需求 4.5**：时间戳以整数微秒表示，解析计算的耗时保留全部亚毫秒精度
  （Property 4：精度不丢失）。

往返一致（Property 1）：对任意合法（B/E 正确嵌套、成对）的事件序列 ``x``，
``serialize_parse_result(parse_systrace(x))`` 与 ``x`` 的有序事件集等价。

内存与体量：逐行流式增量读取已清洗工件（大日志留在磁盘），仅在内存中维护每线程的
未闭合 begin 栈用于配对，避免一次性载入整份文件（design.md「流式解析」机制 4）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from embedded_device_agent.core.models import (
    ArtifactRef,
    CallTreeNode,
    FilterResult,
    ParseResult,
    SystraceEvent,
)

__all__ = ["parse_systrace", "iter_systrace_events", "serialize_parse_result"]


# 匹配一条已清洗 systrace 事件行，捕获时间戳（秒 / 小数）与 B/E|tid|func。
# 形如：``LVGL-1 [0] 2892.002993: tracing_mark_write: B|1|lv_timer_handler``
_EVENT_RE = re.compile(
    r"^\s*\S+-\d+\s+\[\d+\]\s+"
    r"(?P<sec>\d+)\.(?P<frac>\d+)\s*:\s+"
    r"tracing_mark_write\s*:\s+"
    r"(?P<kind>[BE])\|(?P<tid>\d+)\|(?P<func>.+?)\s*$"
)

# systrace 头部 / 注释行——跳过。
_COMMENT_RE = re.compile(r"^\s*#")


def _parse_ts_us(sec: str, frac: str) -> int:
    """把「秒.小数」时间戳解析为整数微秒，保留亚毫秒精度（需求 4.5）。

    小数部分按微秒对齐：右侧补零 / 截断到 6 位。例如 ``2892.002993`` →
    ``2892 * 1_000_000 + 002993`` = ``2892002993`` 微秒；``12.5`` → ``12500000``。
    """
    frac_us = int((frac + "000000")[:6])
    return int(sec) * 1_000_000 + frac_us


def _resolve_path(trace: str | Path | ArtifactRef | FilterResult) -> Path:
    """把多种输入形态归一为已清洗 systrace 文件路径。

    接受直接路径、``ArtifactRef``（``kind == "systrace"``）或 ``Trace_Filter`` 产出的
    ``FilterResult``（取其 ``clean_path``），以便与调用方（Orchestrator）解耦。
    """
    if isinstance(trace, FilterResult):
        return Path(trace.clean_path)
    if isinstance(trace, ArtifactRef):
        return Path(trace.path)
    return Path(trace)


def iter_systrace_events(
    trace: str | Path | ArtifactRef | FilterResult,
) -> Iterator[SystraceEvent]:
    """逐行流式解析已清洗 systrace 工件，产出 ``SystraceEvent`` 事件流。

    以生成器方式增量读取，任一时刻仅一行进入内存，适配数 MB / 几十万事件的大文件
    （design.md「流式解析」机制 4）。注释 / 空行与无法解析的行被跳过。
    """
    path = _resolve_path(trace)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip() or _COMMENT_RE.match(line):
                continue
            m = _EVENT_RE.match(line)
            if m is None:
                continue
            yield SystraceEvent(
                kind=m.group("kind"),
                tid=int(m.group("tid")),
                func=m.group("func"),
                ts_us=_parse_ts_us(m.group("sec"), m.group("frac")),
            )


def parse_systrace(
    trace: str | Path | ArtifactRef | FilterResult | Iterator[SystraceEvent],
) -> ParseResult:
    """把已清洗 systrace 解析为按线程调用树（需求 4.1–4.3, 4.5）。

    Args:
        trace: 已清洗 systrace 工件的路径 / ``ArtifactRef`` / ``FilterResult``，
            或一个已就绪的 ``SystraceEvent`` 事件迭代器（便于单测与往返验证）。

    Returns:
        ``ParseResult``：按 tid 分组的调用树（``trees_by_tid``）与未匹配事件列表
        （``unmatched``，已排除出调用树）。

    配对算法（每线程一栈，维持正确嵌套）：

    * ``B``：以 ``begin_us`` 建节点并压栈（暂不定夺，待匹配 ``E``）。
    * ``E``：若栈顶为同名（同 ``func``）未闭合 ``B``，则闭合成对——写 ``end_us``、
      ``duration_us = end_us - begin_us``，并挂到新栈顶的子节点下（栈空则作为该线程
      的根节点）；否则此 ``E`` 无匹配 begin，记入 ``unmatched`` 并排除。
    * 收尾时仍留在栈中的 ``B``（无对应 ``E``）→ 记入 ``unmatched`` 并排除（需求 4.3）。

    不变量：``duration_us >= 0`` 且子节点时间区间含于父节点区间内（Property 3）；
    耗时精度不低于输入时间戳精度（Property 4）。
    """
    events = trace if _is_event_iter(trace) else iter_systrace_events(trace)  # type: ignore[arg-type]

    trees_by_tid: dict[int, list[CallTreeNode]] = {}
    stacks: dict[int, list[CallTreeNode]] = {}
    unmatched: list[SystraceEvent] = []

    for ev in events:
        roots = trees_by_tid.setdefault(ev.tid, [])
        stack = stacks.setdefault(ev.tid, [])

        if ev.kind == "B":
            stack.append(
                CallTreeNode(
                    func=ev.func,
                    tid=ev.tid,
                    begin_us=ev.ts_us,
                    end_us=ev.ts_us,
                    duration_us=0,
                    children=[],
                )
            )
            continue

        # kind == "E"：仅当栈顶为同名未闭合 begin 时成对闭合。
        if stack and stack[-1].func == ev.func:
            node = stack.pop()
            node.end_us = ev.ts_us
            node.duration_us = ev.ts_us - node.begin_us
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
        else:
            # 无匹配 begin 的孤立 end：报告并排除（需求 4.3）。
            unmatched.append(ev)

    # 收尾：任何仍未闭合的 begin 均无匹配 end → 报告并排除（需求 4.3）。
    for tid, stack in stacks.items():
        for node in stack:
            unmatched.append(
                SystraceEvent(kind="B", tid=tid, func=node.func, ts_us=node.begin_us)
            )

    # 移除仅因未闭合 begin 而产生的空线程条目，保持结果精炼、确定。
    trees_by_tid = {tid: roots for tid, roots in trees_by_tid.items() if roots}

    # 未匹配事件按（时间戳, tid, kind）排序，保证确定性输出。
    unmatched.sort(key=lambda e: (e.ts_us, e.tid, e.kind))

    return ParseResult(trees_by_tid=trees_by_tid, unmatched=unmatched)


def serialize_parse_result(result: ParseResult) -> list[SystraceEvent]:
    """把调用树序列化回有序 ``SystraceEvent`` 事件集（往返，Property 1）。

    对每个节点先发 ``B``（``begin_us``）、递归其子节点、再发 ``E``（``end_us``）——
    该前序遍历对正确嵌套的调用树天然产出按时间戳有序的事件流。跨线程合并后按
    （时间戳, tid）稳定排序，得到与原始输入等价的有序事件集。
    """
    events: list[SystraceEvent] = []

    def _emit(node: CallTreeNode) -> None:
        events.append(
            SystraceEvent(kind="B", tid=node.tid, func=node.func, ts_us=node.begin_us)
        )
        for child in node.children:
            _emit(child)
        events.append(
            SystraceEvent(kind="E", tid=node.tid, func=node.func, ts_us=node.end_us)
        )

    for roots in result.trees_by_tid.values():
        for root in roots:
            _emit(root)

    events.sort(key=lambda e: (e.ts_us, e.tid))
    return events


def _is_event_iter(trace: object) -> bool:
    """判定入参是否已是 ``SystraceEvent`` 事件迭代器（而非路径 / 引用）。"""
    if isinstance(trace, (str, Path, ArtifactRef, FilterResult)):
        return False
    return hasattr(trace, "__iter__") or hasattr(trace, "__next__")
