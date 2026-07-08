"""任务 8.8：``query_trace`` 按需下钻检索的单元测试 + Property 12 覆盖。

覆盖内容（对照 design.md「大日志上下文预算」机制 3 与 trace_query.py 坐标语义）：

* 按坐标取片：``frame_index`` / ``func`` / ``time_window_us`` / ``tid`` 选择，
  以及多坐标 AND 语义；越界 ``frame_index`` 返回空切片。
* ``max_nodes`` 预算裁剪：超预算时置 ``truncated=True`` 并只保留前序前缀。
* **Property 12：原始日志不进上下文**——``query_trace`` 只按需返回**某一小段**
  切片，绝不整体返回调用树；细节始终按坐标分片获取。

输入直接由 ``CallTreeNode`` 构造 ``ParseResult`` 并传入 ``query_trace``，
避免磁盘往返（``query_trace`` 亦接受就绪的 ``ParseResult``）。

**Validates: Requirements 4.1, 5.2**
"""

from __future__ import annotations

from embedded_device_agent.capabilities.perf_tuning.tools.trace_query import (
    query_trace,
)
from embedded_device_agent.core.models import (
    CallTreeNode,
    ParseResult,
    TraceQuery,
    TraceSlice,
)


# --------------------------------------------------------------------------- #
# 辅助构造：确定帧序（derive_frames 按 (begin_us, tid, func) 稳定排序）
#   frame 0 -> tid 1 "lv_timer_handler"(0-1000) → "lv_refr_now"(10-500)
#                                                    → "lv_draw"(20-300)
#   frame 1 -> tid 2 "render"(100-800) → "flush"(150-400)
# --------------------------------------------------------------------------- #
def _node(func, tid, begin, end, children=None):
    return CallTreeNode(
        func=func,
        tid=tid,
        begin_us=begin,
        end_us=end,
        duration_us=end - begin,
        children=children or [],
    )


def _sample_tree() -> ParseResult:
    frame0 = _node(
        "lv_timer_handler",
        1,
        0,
        1000,
        children=[
            _node(
                "lv_refr_now",
                1,
                10,
                500,
                children=[_node("lv_draw", 1, 20, 300)],
            )
        ],
    )
    frame1 = _node(
        "render",
        2,
        100,
        800,
        children=[_node("flush", 2, 150, 400)],
    )
    return ParseResult(trees_by_tid={1: [frame0], 2: [frame1]})


def _count(nodes: list[CallTreeNode]) -> int:
    """切片中节点总数（含所有子孙）。"""
    return sum(1 + _count(n.children) for n in nodes)


def _funcs(nodes: list[CallTreeNode]) -> set[str]:
    out: set[str] = set()
    for n in nodes:
        out.add(n.func)
        out |= _funcs(n.children)
    return out


# --------------------------------------------------------------------------- #
# 按坐标取片
# --------------------------------------------------------------------------- #
def test_select_by_frame_index_returns_that_frame_subtree():
    result = query_trace(_sample_tree(), TraceQuery(frame_index=0))
    assert isinstance(result, TraceSlice)
    assert len(result.nodes) == 1
    assert result.nodes[0].func == "lv_timer_handler"
    # 整棵帧子树（3 个节点）被取回，未被截断。
    assert _count(result.nodes) == 3
    assert result.truncated is False


def test_select_second_frame_by_index():
    result = query_trace(_sample_tree(), TraceQuery(frame_index=1))
    assert len(result.nodes) == 1
    assert result.nodes[0].func == "render"
    assert _funcs(result.nodes) == {"render", "flush"}


def test_select_by_func_locates_topmost_matching_subtree():
    result = query_trace(_sample_tree(), TraceQuery(func="lv_draw"))
    assert _count(result.nodes) == 1
    assert result.nodes[0].func == "lv_draw"
    assert result.truncated is False


def test_select_by_time_window_narrows_within_frame():
    # 仅 lv_draw(20-300) 完全落入窗口；祖先节点区间超出，不应整帧返回。
    result = query_trace(_sample_tree(), TraceQuery(time_window_us=(20, 300)))
    assert _funcs(result.nodes) == {"lv_draw"}


def test_select_by_tid_filters_frames():
    result = query_trace(_sample_tree(), TraceQuery(tid=2))
    assert all(n.tid == 2 for n in result.nodes)
    assert _funcs(result.nodes) == {"render", "flush"}


def test_combined_coordinates_use_and_semantics():
    # frame 0 属 tid 1；要求 tid=2 → 交集为空。
    result = query_trace(_sample_tree(), TraceQuery(frame_index=0, tid=2))
    assert result.nodes == []
    # frame 0 内定位 func=lv_refr_now，取其子树（含 lv_draw）。
    result2 = query_trace(_sample_tree(), TraceQuery(frame_index=0, func="lv_refr_now"))
    assert _funcs(result2.nodes) == {"lv_refr_now", "lv_draw"}


def test_out_of_range_frame_index_returns_empty():
    tree = _sample_tree()
    assert query_trace(tree, TraceQuery(frame_index=99)).nodes == []
    assert query_trace(tree, TraceQuery(frame_index=-1)).nodes == []


def test_unknown_func_returns_empty_slice():
    result = query_trace(_sample_tree(), TraceQuery(func="does_not_exist"))
    assert result.nodes == []
    assert result.truncated is False


# --------------------------------------------------------------------------- #
# max_nodes 预算裁剪
# --------------------------------------------------------------------------- #
def test_max_nodes_truncates_and_sets_flag():
    # frame 0 子树共 3 个节点，预算 2 → 前序保留前 2 个并标记截断。
    result = query_trace(_sample_tree(), TraceQuery(frame_index=0, max_nodes=2))
    assert _count(result.nodes) == 2
    assert result.truncated is True
    # 前序前缀：lv_timer_handler → lv_refr_now。
    assert result.nodes[0].func == "lv_timer_handler"
    assert _funcs(result.nodes) == {"lv_timer_handler", "lv_refr_now"}


def test_max_nodes_exact_budget_not_truncated():
    result = query_trace(_sample_tree(), TraceQuery(frame_index=0, max_nodes=3))
    assert _count(result.nodes) == 3
    assert result.truncated is False


def test_max_nodes_zero_returns_nothing_but_marks_truncated():
    result = query_trace(_sample_tree(), TraceQuery(frame_index=0, max_nodes=0))
    assert result.nodes == []
    assert result.truncated is True


# --------------------------------------------------------------------------- #
# Property 12：原始日志不进上下文——只按需返回一小段切片，绝不整体返回调用树
# --------------------------------------------------------------------------- #
def test_property12_returns_bounded_slice_not_whole_tree():
    # 构造一棵较大的多帧调用树。
    frames_by_tid: dict[int, list[CallTreeNode]] = {}
    total_nodes = 0
    for tid in range(1, 6):
        leaves = [
            _node(f"leaf_{tid}_{j}", tid, 10 * j, 10 * j + 5) for j in range(20)
        ]
        root = _node(f"root_{tid}", tid, 0, 10_000, children=leaves)
        frames_by_tid[tid] = [root]
        total_nodes += 1 + len(leaves)
    tree = ParseResult(trees_by_tid=frames_by_tid)

    # 仅取回单帧的极小片段。
    result = query_trace(tree, TraceQuery(frame_index=0, max_nodes=3))

    returned = _count(result.nodes)
    # 返回受 max_nodes 严格约束，远小于整棵树的节点总数。
    assert returned <= 3
    assert returned < total_nodes
    assert result.truncated is True
    # 绝不整体返回所有线程的调用树：只见到被查询的那一帧根。
    assert len(result.nodes) == 1
    assert result.nodes[0].func == "root_1"


def test_property12_detail_fetched_on_demand_per_coordinate():
    # 每次仅按坐标取回目标一小段，不会牵出其它帧/线程的节点。
    tree = _sample_tree()
    slice1 = query_trace(tree, TraceQuery(func="flush"))
    assert _funcs(slice1.nodes) == {"flush"}
    # 另一坐标独立取片，互不牵连。
    slice2 = query_trace(tree, TraceQuery(func="lv_draw"))
    assert _funcs(slice2.nodes) == {"lv_draw"}


def test_query_trace_is_pure_does_not_mutate_input():
    tree = _sample_tree()
    before = tree.model_dump_json()
    query_trace(tree, TraceQuery(frame_index=0, max_nodes=1))
    assert tree.model_dump_json() == before
