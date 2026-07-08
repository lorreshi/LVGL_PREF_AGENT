"""任务 1.2：共享数据契约的基本构造 + JSON round-trip 覆盖。

仅验证每个模型可独立构造且 JSON 可无损往返；Property 9 的火焰图字段
完整性等更完整覆盖由任务 1.3 实现，此处不重复。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from embedded_device_agent.core.models import (
    ArtifactRef,
    CallTreeNode,
    FilterResult,
    FrameReport,
    HotspotEntry,
    KnowledgeRecord,
    ParseResult,
    RawTraceArtifact,
    SlowFrame,
    SystraceEvent,
    TraceQuery,
    TraceSlice,
)


def _assert_json_roundtrip(model):
    """序列化再反序列化应还原为等价对象。"""
    dumped = model.model_dump_json()
    restored = type(model).model_validate_json(dumped)
    assert restored == model


def test_artifact_ref_roundtrip():
    ref = ArtifactRef(
        run_id="run-1", kind="call_tree", path=Path("/tmp/tree.json"), size_bytes=1024
    )
    _assert_json_roundtrip(ref)


def test_raw_trace_artifact_roundtrip():
    art = RawTraceArtifact(
        run_id="run-1",
        path=Path("/tmp/raw.log"),
        captured_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        duration_s=3.5,
        baud=921600,
    )
    _assert_json_roundtrip(art)


def test_systrace_event_microsecond_precision():
    ev = SystraceEvent(kind="B", tid=7, func="lv_refr_now", ts_us=1234567)
    # 微秒精度必须无损保留
    assert ev.ts_us == 1234567
    _assert_json_roundtrip(ev)


def test_filter_result_roundtrip():
    res = FilterResult(
        clean_path=Path("/tmp/clean.systrace"),
        retained=100,
        excluded=4,
        corrupted_regions=[(10, 12), (40, 41)],
    )
    _assert_json_roundtrip(res)


def test_call_tree_node_nested_roundtrip():
    node = CallTreeNode(
        func="parent",
        tid=1,
        begin_us=0,
        end_us=100,
        duration_us=100,
        children=[
            CallTreeNode(func="child", tid=1, begin_us=10, end_us=40, duration_us=30)
        ],
    )
    _assert_json_roundtrip(node)


def test_parse_result_roundtrip():
    res = ParseResult(
        trees_by_tid={
            1: [CallTreeNode(func="f", tid=1, begin_us=0, end_us=5, duration_us=5)]
        },
        unmatched=[SystraceEvent(kind="B", tid=2, func="g", ts_us=99)],
    )
    _assert_json_roundtrip(res)


def test_hotspot_entry_roundtrip():
    hs = HotspotEntry(func="lv_draw", total_us=5000, call_count=12, rank=1)
    _assert_json_roundtrip(hs)


def test_slow_frame_roundtrip():
    sf = SlowFrame(index=3, duration_us=25000, dominant_func="lv_draw")
    _assert_json_roundtrip(sf)


def test_frame_report_roundtrip():
    report = FrameReport(
        scenario="scroll_list",
        target_fps=60,
        frame_budget_us=16667,
        slow_frames=[SlowFrame(index=1, duration_us=30000, dominant_func="lv_draw")],
        hotspots=[HotspotEntry(func="lv_draw", total_us=30000, call_count=1, rank=1)],
        total_frames=120,
        p95_frame_us=18000,
        source=ArtifactRef(
            run_id="run-1", kind="call_tree", path=Path("/tmp/t.json"), size_bytes=42
        ),
        summary="draw dominates",
    )
    assert report.no_slow_frames is False
    _assert_json_roundtrip(report)


def test_trace_query_roundtrip():
    q = TraceQuery(frame_index=3, func="lv_draw", tid=1, time_window_us=(0, 1000))
    _assert_json_roundtrip(q)


def test_trace_query_defaults():
    q = TraceQuery()
    assert q.max_nodes == 200
    assert q.frame_index is None


def test_trace_slice_roundtrip():
    sl = TraceSlice(
        nodes=[CallTreeNode(func="f", tid=1, begin_us=0, end_us=5, duration_us=5)],
        truncated=True,
    )
    _assert_json_roundtrip(sl)


def test_knowledge_record_roundtrip():
    rec = KnowledgeRecord(
        symptom="scroll jank",
        root_cause="synchronous decode",
        optimization="cache decoded image",
        effect="p95 18ms -> 12ms",
        scenario="scroll_list",
        created_at=datetime(2024, 6, 1, 9, 30, tzinfo=timezone.utc),
    )
    assert rec.id is None
    _assert_json_roundtrip(rec)
