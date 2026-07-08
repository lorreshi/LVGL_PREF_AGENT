"""Trace_Filter：原始 LVGL profiler 日志 → 已清洗 systrace 工件（任务 8.1）。

封装 LVGL `scripts/trace_filter.py` 的核心职责，作为**确定性普通工具**由
Orchestrator 调用（需求 13.2）。纯输入输出：给定相同原始日志内容，恒产生相同的
清洗结果与计数，除写出已清洗工件外不产生其他外部副作用（Property 8）。

LVGL 内置 profiler 以 Android systrace 格式输出事件行，形如::

    LVGL-1 [0] 2892.002993: tracing_mark_write: B|1|lv_timer_handler
    LVGL-1 [0] 2892.003550: tracing_mark_write: E|1|lv_draw_rect

其中时间戳为「秒.微秒」（亚毫秒精度），``B`` / ``E`` 为一对函数进入 / 退出标记，
``tid`` 为线程号，``func`` 为函数名。当串口波特率过高或其他线程日志在打印期间
插入时，会出现**交错**——破坏 begin/end 配对、损坏数据（见 LVGL profiler FAQ）。

本工具行为（严格对照需求 3）：

* **需求 3.1**：把原始日志转换为 Android systrace 格式，产出一个已清洗 trace 工件
  （写到 ``FilterResult.clean_path``）。
* **需求 3.2**：对破坏 begin/end 配对的交错输出，报告受损区间
  （``corrupted_regions``：原始日志中的行号区间），并将未配对事件排除出已清洗工件。
* **需求 3.3**：报告被保留与被排除的 Systrace_Event 数量（``retained`` / ``excluded``）。
* **需求 3.4**：在已清洗工件中保留每个被保留事件的时间戳（微秒精度）。

关键不变量（Property 2：B/E 配对守恒）：``retained + excluded`` 等于从原始日志中
解析出的事件总数；被保留的事件必然成对——每个 ``B`` 都有同 ``tid``、时间戳不早于它
的对应 ``E``，且嵌套正确。

内存与体量：逐行流式读取原始日志（大日志留在磁盘），仅在内存中保留轻量事件记录
用于配对判定，再将保留事件按原始顺序写出。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from embedded_device_agent.core.models import FilterResult, RawTraceArtifact

__all__ = ["trace_filter"]


# 匹配一条 LVGL systrace 事件行，捕获时间戳（秒 / 小数部分）与 B/E|tid|func。
# 形如：``LVGL-1 [0] 2892.002993: tracing_mark_write: B|1|lv_timer_handler``
_EVENT_RE = re.compile(
    r"^\s*\S+-\d+\s+\[\d+\]\s+"
    r"(?P<sec>\d+)\.(?P<frac>\d+)\s*:\s+"
    r"tracing_mark_write\s*:\s+"
    r"(?P<kind>[BE])\|(?P<tid>\d+)\|(?P<func>.+?)\s*$"
)

# systrace 头部 / 注释行——跳过，不计为事件也不计为损坏。
_COMMENT_RE = re.compile(r"^\s*#")


def _parse_ts_us(sec: str, frac: str) -> int:
    """把「秒.小数」时间戳解析为整数微秒，保留亚毫秒精度（需求 3.4 / 4.5）。

    小数部分按微秒对齐：右侧补零 / 截断到 6 位。例如 ``2892.002993`` →
    ``2892 * 1_000_000 + 002993`` = ``2892002993`` 微秒；``12.5`` → ``12500000``。
    """
    frac_us = int((frac + "000000")[:6])
    return int(sec) * 1_000_000 + frac_us


def _format_event(kind: str, tid: int, func: str, ts_us: int) -> str:
    """把一个被保留事件格式化回 systrace 行，无损保留其时间戳（需求 3.4）。"""
    sec, frac_us = divmod(ts_us, 1_000_000)
    return f"LVGL-{tid} [0] {sec}.{frac_us:06d}: tracing_mark_write: {kind}|{tid}|{func}"


@dataclass
class _Event:
    """解析出的候选事件，附带其在原始日志中的行号（用于报告受损区间）。"""

    kind: str  # "B" / "E"
    tid: int
    func: str
    ts_us: int
    line_no: int
    retained: bool = False


def _pair_events(events: list[_Event]) -> None:
    """按 tid 用栈匹配 B/E 配对，就地把成对事件标记为 ``retained``（Property 2）。

    对每个线程维护一个未闭合 ``B`` 事件栈：

    * 遇到 ``B``：压栈（暂不保留，待匹配到对应 ``E`` 再定夺）。
    * 遇到 ``E``：从栈顶向下寻找最近的同名 ``B``：
        - 找到 → 该 ``B`` 与此 ``E`` 成对保留；位于其上、因交错而未闭合的内层
          ``B`` 判为破坏配对，保持未保留（将被排除）。
        - 找不到（无匹配 ``B`` 的孤立 ``E``）→ 此 ``E`` 破坏配对，保持未保留。
    * 结束时仍留在栈中的 ``B``（无对应 ``E``）→ 破坏配对，保持未保留。

    该「向下搜索最近匹配」策略在丢失中间事件的交错场景下，尽量保留完好的外层配对，
    仅排除真正被破坏的事件。被保留事件因此必然成对且嵌套正确。
    """
    stacks: dict[int, list[_Event]] = {}
    for ev in events:
        stack = stacks.setdefault(ev.tid, [])
        if ev.kind == "B":
            stack.append(ev)
            continue
        # kind == "E"：向下寻找最近的同名未闭合 B。
        match_idx: int | None = None
        for i in range(len(stack) - 1, -1, -1):
            if stack[i].func == ev.func:
                match_idx = i
                break
        if match_idx is None:
            # 孤立 E：无匹配 B，破坏配对 → 排除（保持 retained=False）。
            continue
        # 匹配成功：外层 B 与此 E 成对保留；其上未闭合的内层 B 保持排除。
        begin = stack[match_idx]
        begin.retained = True
        ev.retained = True
        del stack[match_idx:]


def _coalesce_regions(lines: list[int]) -> list[tuple[int, int]]:
    """把离散的受损行号合并为连续的 ``(起, 止)`` 闭区间列表（需求 3.2）。"""
    if not lines:
        return []
    ordered = sorted(set(lines))
    regions: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for ln in ordered[1:]:
        if ln == prev + 1:
            prev = ln
            continue
        regions.append((start, prev))
        start = prev = ln
    regions.append((start, prev))
    return regions


def trace_filter(
    raw: RawTraceArtifact,
    *,
    clean_path: str | Path | None = None,
) -> FilterResult:
    """把原始 LVGL profiler 日志清洗为 systrace 工件（需求 3.1–3.4）。

    Args:
        raw: 指向磁盘上原始日志文件的采集工件（``raw.path``）。
        clean_path: 已清洗 systrace 工件的写出路径；缺省时置于原始日志同目录、
            以 ``.systrace`` 为后缀（可注入以便测试指定输出位置）。

    Returns:
        ``FilterResult``：已清洗工件路径、保留 / 排除计数与受损区间。

    不变量：``retained + excluded`` == 原始日志中解析出的事件总数；被保留事件必然
    成对（Property 2）。相同输入内容恒产生相同输出（Property 8）。
    """
    raw_path = Path(raw.path)
    out_path = (
        Path(clean_path)
        if clean_path is not None
        else raw_path.with_name(raw_path.stem + ".systrace")
    )

    events: list[_Event] = []
    corrupted_lines: list[int] = []

    # 逐行流式读取原始日志：识别事件行、跳过注释、记录无法解析的交错噪声行。
    with raw_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip() or _COMMENT_RE.match(line):
                continue
            m = _EVENT_RE.match(line)
            if m is None:
                # 无法解析的行：其他线程插入的交错输出 / 串口误码 → 受损区间。
                corrupted_lines.append(line_no)
                continue
            events.append(
                _Event(
                    kind=m.group("kind"),
                    tid=int(m.group("tid")),
                    func=m.group("func"),
                    ts_us=_parse_ts_us(m.group("sec"), m.group("frac")),
                    line_no=line_no,
                )
            )

    # 按 tid 配对，标记成对保留的事件（破坏配对者保持未保留 → 排除）。
    _pair_events(events)

    retained_events = [ev for ev in events if ev.retained]
    excluded_events = [ev for ev in events if not ev.retained]

    # 受损区间 = 无法解析的噪声行 + 因破坏配对而被排除的事件行，合并为连续区间。
    corrupted_lines.extend(ev.line_no for ev in excluded_events)
    corrupted_regions = _coalesce_regions(corrupted_lines)

    # 写出已清洗 systrace 工件：标准头部 + 按原始顺序保留、无损时间戳的事件行。
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# tracer: nop\n#\n")
        for ev in retained_events:
            fh.write(_format_event(ev.kind, ev.tid, ev.func, ev.ts_us) + "\n")

    return FilterResult(
        clean_path=out_path,
        retained=len(retained_events),
        excluded=len(excluded_events),
        corrupted_regions=corrupted_regions,
    )
