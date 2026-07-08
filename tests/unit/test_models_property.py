"""任务 1.3：数据契约的校验 + Property 9（工件语言中立）覆盖。

本文件在任务 1.2 的基本 round-trip 之上补齐两块：

1. **模型校验**：非法枚举值 / 缺失必填字段应被 Pydantic v2 拒绝。
2. **Property 9 —— 工件语言中立**（design.md）：经 API 边界暴露的报告工件
   （``FrameReport`` / ``SlowFrame`` / ``HotspotEntry``，以及承载线程/时间戳的
   ``CallTreeNode``）可无损 JSON 序列化，且序列化结果保留渲染火焰图或慢帧
   报告所需的字段——函数名、时间戳、耗时、线程标识。
   用 hypothesis 在大量随机输入上验证该属性。

**Validates: Requirements 15.3, 15.4**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from embedded_device_agent.core.models import (
    ArtifactRef,
    CallTreeNode,
    FrameReport,
    HotspotEntry,
    SlowFrame,
)

# --------------------------------------------------------------------------- #
# 生成器：约束到工件的合理输入空间（火焰图字段均为非负整数/非空标识符）
# --------------------------------------------------------------------------- #

# 函数名：含 LVGL 风格标识符与部分 Unicode，但排除代理项以保证 JSON 合法。
_func_names = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=0x2FFF, blacklist_categories=("Cs",)),
    min_size=1,
    max_size=48,
)
# 安全的 id / 路径片段（字母数字 + 下划线/连字符），避免 Path 归一化带来的等价性歧义。
_safe_tokens = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"),
    min_size=1,
    max_size=24,
)
_us_values = st.integers(min_value=0, max_value=10**12)  # 微秒时间戳/耗时
_tids = st.integers(min_value=0, max_value=4096)
_counts = st.integers(min_value=0, max_value=100_000)


@st.composite
def _slow_frames(draw: st.DrawFn) -> SlowFrame:
    return SlowFrame(
        index=draw(st.integers(min_value=0, max_value=100_000)),
        duration_us=draw(_us_values),
        dominant_func=draw(_func_names),
    )


@st.composite
def _hotspot_entries(draw: st.DrawFn) -> HotspotEntry:
    return HotspotEntry(
        func=draw(_func_names),
        total_us=draw(_us_values),
        call_count=draw(_counts),
        rank=draw(st.integers(min_value=0, max_value=1000)),
    )


@st.composite
def _call_tree_nodes(draw: st.DrawFn, max_depth: int = 3) -> CallTreeNode:
    begin = draw(_us_values)
    duration = draw(_us_values)
    children: list[CallTreeNode] = []
    if max_depth > 0:
        children = draw(
            st.lists(_call_tree_nodes(max_depth=max_depth - 1), max_size=2)
        )
    return CallTreeNode(
        func=draw(_func_names),
        tid=draw(_tids),
        begin_us=begin,
        end_us=begin + duration,
        duration_us=duration,
        children=children,
    )


@st.composite
def _artifact_refs(draw: st.DrawFn) -> ArtifactRef:
    return ArtifactRef(
        run_id=draw(_safe_tokens),
        kind=draw(st.sampled_from(["raw_log", "systrace", "call_tree"])),
        path=Path("/tmp") / draw(_safe_tokens),
        size_bytes=draw(st.integers(min_value=0, max_value=10**9)),
    )


@st.composite
def _frame_reports(draw: st.DrawFn) -> FrameReport:
    return FrameReport(
        scenario=draw(_safe_tokens),
        target_fps=draw(st.integers(min_value=1, max_value=240)),
        frame_budget_us=draw(st.integers(min_value=1, max_value=10**7)),
        slow_frames=draw(st.lists(_slow_frames(), max_size=5)),
        hotspots=draw(st.lists(_hotspot_entries(), max_size=5)),
        total_frames=draw(_counts),
        p95_frame_us=draw(_us_values),
        source=draw(_artifact_refs()),
        summary=draw(st.none() | _func_names),
        no_slow_frames=draw(st.booleans()),
    )


def _roundtrip(model):
    """JSON dump → validate 应还原等价对象；返回反序列化后的 dict 以便断言字段。"""
    dumped = model.model_dump_json()
    restored = type(model).model_validate_json(dumped)
    assert restored == model
    return json.loads(dumped)


# --------------------------------------------------------------------------- #
# 模型校验（构造 / 校验）
# --------------------------------------------------------------------------- #


def test_artifact_ref_rejects_invalid_kind():
    with pytest.raises(ValidationError):
        ArtifactRef(run_id="r", kind="bogus", path="/tmp/x", size_bytes=1)


def test_systrace_event_like_missing_required_field_rejected():
    # SlowFrame 缺少必填 dominant_func 应报错
    with pytest.raises(ValidationError):
        SlowFrame(index=1, duration_us=10)  # type: ignore[call-arg]


def test_hotspot_entry_type_coercion_rejects_non_int():
    with pytest.raises(ValidationError):
        HotspotEntry(func="f", total_us="not-an-int", call_count=1, rank=1)  # type: ignore[arg-type]


def test_frame_report_requires_source_and_aggregates():
    with pytest.raises(ValidationError):
        # 缺少必填的 source / total_frames / p95_frame_us
        FrameReport(scenario="s", target_fps=60, frame_budget_us=16667)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Property 9：工件语言中立（hypothesis 属性测试）
# **Validates: Requirements 15.3, 15.4**
# --------------------------------------------------------------------------- #


@settings(max_examples=200, deadline=None)
@given(sf=_slow_frames())
def test_property9_slow_frame_language_neutral(sf: SlowFrame):
    """SlowFrame 无损 round-trip，且保留函数名与耗时字段。"""
    data = _roundtrip(sf)
    assert data["dominant_func"] == sf.dominant_func  # 函数名
    assert data["duration_us"] == sf.duration_us  # 耗时
    assert data["index"] == sf.index


@settings(max_examples=200, deadline=None)
@given(hs=_hotspot_entries())
def test_property9_hotspot_entry_language_neutral(hs: HotspotEntry):
    """HotspotEntry 无损 round-trip，且保留函数名与聚合耗时字段。"""
    data = _roundtrip(hs)
    assert data["func"] == hs.func  # 函数名
    assert data["total_us"] == hs.total_us  # 聚合耗时
    assert data["call_count"] == hs.call_count
    assert data["rank"] == hs.rank


@settings(max_examples=200, deadline=None)
@given(node=_call_tree_nodes())
def test_property9_call_tree_node_preserves_flame_fields(node: CallTreeNode):
    """CallTreeNode（火焰图渲染单元）保留函数名、时间戳、耗时、线程标识。"""
    data = _roundtrip(node)
    assert data["func"] == node.func  # 函数名
    assert data["tid"] == node.tid  # 线程标识
    assert data["begin_us"] == node.begin_us  # 时间戳（起）
    assert data["end_us"] == node.end_us  # 时间戳（止）
    assert data["duration_us"] == node.duration_us  # 耗时
    # 耗时精度不因序列化而丢失（微秒整数无损）
    assert isinstance(data["begin_us"], int) and isinstance(data["duration_us"], int)


@settings(max_examples=150, deadline=None)
@given(report=_frame_reports())
def test_property9_frame_report_language_neutral(report: FrameReport):
    """FrameReport 无损 round-trip，且其嵌套慢帧/热点保留火焰图字段。"""
    data = _roundtrip(report)
    # 嵌套慢帧：函数名 + 耗时逐条保留
    assert len(data["slow_frames"]) == len(report.slow_frames)
    for dumped_sf, sf in zip(data["slow_frames"], report.slow_frames):
        assert dumped_sf["dominant_func"] == sf.dominant_func
        assert dumped_sf["duration_us"] == sf.duration_us
    # 嵌套热点：函数名 + 聚合耗时逐条保留
    assert len(data["hotspots"]) == len(report.hotspots)
    for dumped_hs, hs in zip(data["hotspots"], report.hotspots):
        assert dumped_hs["func"] == hs.func
        assert dumped_hs["total_us"] == hs.total_us
    # 聚合统计与下钻引用保留
    assert data["total_frames"] == report.total_frames
    assert data["p95_frame_us"] == report.p95_frame_us
    assert data["source"]["run_id"] == report.source.run_id


def test_property9_frame_report_json_is_pure_json_types():
    """报告序列化结果应为纯 JSON 原生类型（语言中立，需求 15.3）。"""
    report = FrameReport(
        scenario="scroll_list",
        target_fps=60,
        frame_budget_us=16667,
        slow_frames=[SlowFrame(index=1, duration_us=30000, dominant_func="lv_draw")],
        hotspots=[HotspotEntry(func="lv_draw", total_us=30000, call_count=1, rank=1)],
        total_frames=120,
        p95_frame_us=18000,
        source=ArtifactRef(
            run_id="run-1", kind="call_tree", path="/tmp/t.json", size_bytes=42
        ),
    )
    data = json.loads(report.model_dump_json())

    def _assert_json_native(value):
        assert isinstance(value, (dict, list, str, int, float, bool, type(None)))
        if isinstance(value, dict):
            for v in value.values():
                _assert_json_native(v)
        elif isinstance(value, list):
            for v in value:
                _assert_json_native(v)

    _assert_json_native(data)
    # 火焰图字段齐备
    assert data["slow_frames"][0]["dominant_func"] == "lv_draw"
    assert data["hotspots"][0]["func"] == "lv_draw"
    assert data["source"]["path"] == "/tmp/t.json"
