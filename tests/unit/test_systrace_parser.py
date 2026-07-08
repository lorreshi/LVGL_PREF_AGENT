"""任务 8.4：Trace_Parser 单元测试 + round-trip / 精度 / 嵌套属性测试。

覆盖 ``systrace_parser`` 三个入口：``parse_systrace``、``iter_systrace_events``、
``serialize_parse_result``（见其 docstring 与需求 4）。

单元测试：
* 正常 B/E 配对为按线程（tid）组织的调用树（需求 4.1/4.2）。
* 缺失 end 的 begin（及孤立 end）被报告并排除出调用树（需求 4.3）。
* 亚毫秒（µs）精度在从 systrace 文本解析的耗时中被完整保留（需求 4.5）。

属性测试（hypothesis，生成合法且正确嵌套的 B/E 事件序列）：
* **Property 1 —— 解析往返一致**：``serialize_parse_result(parse_systrace(x))``
  与 ``x`` 的有序事件集等价（需求 4.4）。
* **Property 3 —— 耗时非负且区间嵌套**：``duration_us >= 0``，且子区间含于父区间
  （需求 4.2）。
* **Property 4 —— 精度不丢失**：耗时为整数微秒的精确差，精度不低于输入时间戳精度
  （需求 4.5）。

**Validates: Requirements 4.4, 4.2, 4.5**
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from embedded_device_agent.capabilities.perf_tuning.tools.systrace_parser import (
    iter_systrace_events,
    parse_systrace,
    serialize_parse_result,
)
from embedded_device_agent.core.models import CallTreeNode, SystraceEvent

# --------------------------------------------------------------------------- #
# 帮助函数
# --------------------------------------------------------------------------- #


def _write_trace(tmp_path: Path, lines: list[str]) -> Path:
    """把 systrace 事件行写入临时文件，返回其路径。"""
    path = tmp_path / "clean.systrace"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _event_tuple(e: SystraceEvent) -> tuple[int, int, str, str]:
    """把事件归一为可比较元组（时间戳, tid, kind, func），用于有序集等价比较。"""
    return (e.ts_us, e.tid, e.kind, e.func)


def _walk(nodes: list[CallTreeNode]):
    """先序遍历调用树，逐个产出节点。"""
    for node in nodes:
        yield node
        yield from _walk(node.children)


# --------------------------------------------------------------------------- #
# 单元测试：正常配对 → 按线程调用树（需求 4.1/4.2）
# --------------------------------------------------------------------------- #


def test_normal_pairing_builds_per_thread_call_tree(tmp_path: Path):
    """嵌套的 B/E 对应构成父子调用树，耗时为 end-begin。"""
    trace = _write_trace(
        tmp_path,
        [
            "LVGL-1 [0] 2892.000000: tracing_mark_write: B|1|lv_timer_handler",
            "LVGL-1 [0] 2892.001000: tracing_mark_write: B|1|lv_refr_area",
            "LVGL-1 [0] 2892.002000: tracing_mark_write: E|1|lv_refr_area",
            "LVGL-1 [0] 2892.004000: tracing_mark_write: E|1|lv_timer_handler",
        ],
    )

    result = parse_systrace(trace)

    assert result.unmatched == []
    assert set(result.trees_by_tid) == {1}
    roots = result.trees_by_tid[1]
    assert len(roots) == 1

    outer = roots[0]
    assert outer.func == "lv_timer_handler"
    assert outer.tid == 1
    assert outer.begin_us == 2892_000_000
    assert outer.end_us == 2892_004_000
    assert outer.duration_us == 4000

    assert len(outer.children) == 1
    inner = outer.children[0]
    assert inner.func == "lv_refr_area"
    assert inner.begin_us == 2892_001_000
    assert inner.end_us == 2892_002_000
    assert inner.duration_us == 1000
    # 子区间含于父区间（需求 4.2 嵌套）
    assert outer.begin_us <= inner.begin_us <= inner.end_us <= outer.end_us


def test_multiple_threads_grouped_by_tid(tmp_path: Path):
    """不同 tid 的事件分别构成各自线程的调用树。"""
    trace = _write_trace(
        tmp_path,
        [
            "LVGL-1 [0] 10.000000: tracing_mark_write: B|1|thread_one",
            "LVGL-2 [1] 10.000100: tracing_mark_write: B|2|thread_two",
            "LVGL-2 [1] 10.000300: tracing_mark_write: E|2|thread_two",
            "LVGL-1 [0] 10.000500: tracing_mark_write: E|1|thread_one",
        ],
    )

    result = parse_systrace(trace)

    assert result.unmatched == []
    assert set(result.trees_by_tid) == {1, 2}
    assert result.trees_by_tid[1][0].func == "thread_one"
    assert result.trees_by_tid[2][0].func == "thread_two"


# --------------------------------------------------------------------------- #
# 单元测试：未匹配事件被报告并排除（需求 4.3）
# --------------------------------------------------------------------------- #


def test_unmatched_begin_missing_end_excluded_and_reported(tmp_path: Path):
    """没有匹配 end 的 begin 被记入 unmatched 并排除出调用树。"""
    trace = _write_trace(
        tmp_path,
        [
            "LVGL-1 [0] 5.000000: tracing_mark_write: B|1|paired",
            "LVGL-1 [0] 5.000100: tracing_mark_write: E|1|paired",
            "LVGL-1 [0] 5.000200: tracing_mark_write: B|1|orphan_begin",
        ],
    )

    result = parse_systrace(trace)

    # 已配对的 paired 作为根保留
    assert [r.func for r in result.trees_by_tid[1]] == ["paired"]
    # orphan_begin 被排除且被报告为未匹配 begin
    assert len(result.unmatched) == 1
    ev = result.unmatched[0]
    assert ev.kind == "B"
    assert ev.func == "orphan_begin"
    assert ev.tid == 1
    # 排除后调用树中不含 orphan_begin
    funcs = {n.func for n in _walk(result.trees_by_tid[1])}
    assert "orphan_begin" not in funcs


def test_unmatched_stray_end_reported(tmp_path: Path):
    """没有匹配 begin 的孤立 end 被记入 unmatched 并排除。"""
    trace = _write_trace(
        tmp_path,
        [
            "LVGL-1 [0] 7.000000: tracing_mark_write: E|1|stray_end",
            "LVGL-1 [0] 7.000100: tracing_mark_write: B|1|ok",
            "LVGL-1 [0] 7.000200: tracing_mark_write: E|1|ok",
        ],
    )

    result = parse_systrace(trace)

    assert [r.func for r in result.trees_by_tid[1]] == ["ok"]
    assert len(result.unmatched) == 1
    assert result.unmatched[0].kind == "E"
    assert result.unmatched[0].func == "stray_end"


# --------------------------------------------------------------------------- #
# 单元测试：亚毫秒（µs）精度保留（需求 4.5）
# --------------------------------------------------------------------------- #


def test_submillisecond_precision_preserved(tmp_path: Path):
    """微秒级时间戳被完整解析，耗时保留亚毫秒精度。"""
    trace = _write_trace(
        tmp_path,
        [
            "LVGL-1 [0] 2892.002993: tracing_mark_write: B|1|lv_timer_handler",
            "LVGL-1 [0] 2892.003550: tracing_mark_write: E|1|lv_timer_handler",
        ],
    )

    result = parse_systrace(trace)
    node = result.trees_by_tid[1][0]

    assert node.begin_us == 2892_002_993
    assert node.end_us == 2892_003_550
    # 557µs——亚毫秒级耗时未被舍入到毫秒
    assert node.duration_us == 557
    assert node.duration_us % 1000 != 0


def test_iter_systrace_events_reads_all_events(tmp_path: Path):
    """iter_systrace_events 逐行产出全部合法事件，跳过注释/空行/杂行。"""
    trace = _write_trace(
        tmp_path,
        [
            "# tracer: nop",
            "",
            "LVGL-1 [0] 1.000000: tracing_mark_write: B|1|foo",
            "some garbage line that does not match",
            "LVGL-1 [0] 1.000500: tracing_mark_write: E|1|foo",
        ],
    )

    events = list(iter_systrace_events(trace))

    assert [(e.kind, e.func, e.ts_us) for e in events] == [
        ("B", "foo", 1_000_000),
        ("E", "foo", 1_000_500),
    ]


# --------------------------------------------------------------------------- #
# hypothesis 策略：生成合法且正确嵌套的 B/E 事件序列
# --------------------------------------------------------------------------- #

# 标识符风格函数名（字母/数字/下划线），保证事件对象干净可比。
_func_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=10,
)


@st.composite
def _nested_events(draw: st.DrawFn) -> list[SystraceEvent]:
    """生成一个正确嵌套、成对的 B/E 事件序列（可跨多个 tid）。

    时间戳由全局单调递增的时钟分配（每步增量为任意正微秒，含非毫秒整数以覆盖
    亚毫秒精度），从而保证：每对 begin/end 满足 ``end_us >= begin_us``、子区间含于
    父区间，且全局时间戳唯一——使往返「有序事件集」比较无歧义。
    """
    n_threads = draw(st.integers(min_value=1, max_value=3))
    tids = draw(
        st.lists(
            st.integers(min_value=0, max_value=99),
            min_size=n_threads,
            max_size=n_threads,
            unique=True,
        )
    )

    clock = [draw(st.integers(min_value=0, max_value=1000))]

    def tick() -> int:
        clock[0] += draw(st.integers(min_value=1, max_value=5000))
        return clock[0]

    events: list[SystraceEvent] = []

    def build_and_emit(tid: int, depth: int) -> None:
        count = draw(st.integers(min_value=0, max_value=3))
        for _ in range(count):
            func = draw(_func_names)
            begin = tick()
            events.append(SystraceEvent(kind="B", tid=tid, func=func, ts_us=begin))
            if depth > 0:
                build_and_emit(tid, depth - 1)
            end = tick()
            events.append(SystraceEvent(kind="E", tid=tid, func=func, ts_us=end))

    for tid in tids:
        # 每线程至少产出一个根调用，保证多数样本非空。
        func = draw(_func_names)
        begin = tick()
        events.append(SystraceEvent(kind="B", tid=tid, func=func, ts_us=begin))
        build_and_emit(tid, draw(st.integers(min_value=0, max_value=2)))
        end = tick()
        events.append(SystraceEvent(kind="E", tid=tid, func=func, ts_us=end))

    return events


# --------------------------------------------------------------------------- #
# Property 1：解析往返一致（需求 4.4）
# **Validates: Requirements 4.4**
# --------------------------------------------------------------------------- #


@settings(max_examples=200, deadline=None)
@given(events=_nested_events())
def test_property1_parse_serialize_roundtrip(events: list[SystraceEvent]):
    """serialize_parse_result(parse_systrace(x)) 与 x 的有序事件集等价。"""
    result = parse_systrace(iter(events))

    # 合法（正确嵌套、成对）输入不应产生任何未匹配事件。
    assert result.unmatched == []

    roundtrip = serialize_parse_result(result)

    # 有序事件集等价：作为 (ts, tid, kind, func) 的有序集比较。
    assert sorted(map(_event_tuple, roundtrip)) == sorted(map(_event_tuple, events))


# --------------------------------------------------------------------------- #
# Property 3：耗时非负且区间嵌套（需求 4.2）
# **Validates: Requirements 4.2**
# --------------------------------------------------------------------------- #


@settings(max_examples=200, deadline=None)
@given(events=_nested_events())
def test_property3_duration_nonneg_and_intervals_nested(events: list[SystraceEvent]):
    """每个节点 duration_us >= 0，且每个子节点区间含于父节点区间。"""
    result = parse_systrace(iter(events))

    def check(node: CallTreeNode) -> None:
        assert node.duration_us >= 0
        assert node.end_us >= node.begin_us
        assert node.duration_us == node.end_us - node.begin_us
        for child in node.children:
            # 子区间嵌套于父区间之内
            assert node.begin_us <= child.begin_us
            assert child.end_us <= node.end_us
            check(child)

    for roots in result.trees_by_tid.values():
        for root in roots:
            check(root)


# --------------------------------------------------------------------------- #
# Property 4：精度不丢失（需求 4.5）
# **Validates: Requirements 4.5**
# --------------------------------------------------------------------------- #


@settings(max_examples=200, deadline=None)
@given(events=_nested_events())
def test_property4_precision_not_lost(events: list[SystraceEvent]):
    """耗时为整数微秒的精确差；节点时间戳与输入事件时间戳逐一相等，无精度损失。"""
    result = parse_systrace(iter(events))

    # 输入中每个 (tid, func) begin/end 的微秒时间戳集合
    begins: dict[tuple[int, str], list[int]] = {}
    ends: dict[tuple[int, str], list[int]] = {}
    for e in events:
        target = begins if e.kind == "B" else ends
        target.setdefault((e.tid, e.func), []).append(e.ts_us)

    for node in (n for roots in result.trees_by_tid.values() for n in _walk(roots)):
        # 耗时精度不低于输入：整数微秒的精确差，无任何舍入。
        assert node.duration_us == node.end_us - node.begin_us
        assert isinstance(node.duration_us, int)
        # 节点起止时间戳来自输入事件，未损失微秒精度。
        assert node.begin_us in begins[(node.tid, node.func)]
        assert node.end_us in ends[(node.tid, node.func)]
