"""任务 11.7：Instrumentor（埋点器）单元测试。

对照需求 6 与 design.md「编排层与智能体层 → Instrumentor」，覆盖三条核心行为：

* **成对埋点插入（需求 6.1）**：对源码区域 ``[begin_line, end_line]`` 插入成对
  ``LV_PROFILER_BEGIN_TAG`` / ``LV_PROFILER_END_TAG``——BEGIN 插到区域首行之前、
  END 插到末行之后，且二者共用同一标签（配对）。
* **字符串字面量标签（需求 6.2）**：标签一律以 C 字符串字面量输出，特殊字符转义。
* **微秒 tick 精度报告分支（需求 6.6）**：默认 tick 分辨率不优于亚毫秒时报告精度
  不足（需引入微秒级 tick 回调），优于时报告充分。

纯编辑函数不触达磁盘/硬件；Instrumentor 类经注入的 ``FakeDeviceIO`` /
``FakeLLMProvider`` 驱动，重编译/烧录副作用离线可重复回放。
"""

from __future__ import annotations

import pytest

from embedded_device_agent.capabilities.perf_tuning.agents.instrumentor import (
    SUBMS_RESOLUTION_US,
    Instrumentor,
    InstrumentationPoint,
    PrecisionReport,
    c_string_literal,
    check_timing_precision,
    insert_instrumentation,
    insert_many,
    render_begin_tag,
    render_end_tag,
)
from embedded_device_agent.core.device.models import BuildResult, FlashResult
from tests.harness import FakeDeviceIO, FakeLLMProvider

# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #

_SOURCE = (
    "void lv_draw(void) {\n"  # 1
    "    int a = 1;\n"        # 2
    "    int b = 2;\n"        # 3
    "    do_work(a, b);\n"    # 4
    "}\n"                     # 5
)


def _point(**overrides) -> InstrumentationPoint:
    base = dict(file="draw.c", tag="draw_body", begin_line=2, end_line=4)
    base.update(overrides)
    return InstrumentationPoint(**base)


# --------------------------------------------------------------------------- #
# 需求 6.1：成对 BEGIN/END 插入 + 配对
# --------------------------------------------------------------------------- #


def test_begin_inserted_before_region_end_after_region():
    """BEGIN 插到区域首行之前、END 插到末行之后，区域内容位置不变（需求 6.1）。"""
    result = insert_instrumentation(_SOURCE, _point(begin_line=2, end_line=4))
    lines = result.splitlines()

    # 原 5 行 + 一对埋点 = 7 行
    assert len(lines) == 7
    # BEGIN 紧跟函数首行、在原区域首行(int a = 1;)之前
    assert lines[0] == "void lv_draw(void) {"
    assert lines[1].strip() == 'LV_PROFILER_BEGIN_TAG("draw_body");'
    assert lines[2].strip() == "int a = 1;"
    # END 紧跟区域末行(do_work)之后、在闭合括号之前
    assert lines[4].strip() == "do_work(a, b);"
    assert lines[5].strip() == 'LV_PROFILER_END_TAG("draw_body");'
    assert lines[6] == "}"


def test_begin_and_end_share_same_tag_for_pairing():
    """成对埋点共用同一标签，保证 BEGIN/END 可配对（需求 6.1）。"""
    result = insert_instrumentation(_SOURCE, _point(tag="hot_region"))
    assert result.count('LV_PROFILER_BEGIN_TAG("hot_region");') == 1
    assert result.count('LV_PROFILER_END_TAG("hot_region");') == 1


def test_inserted_tags_preserve_region_indentation():
    """插入的埋点沿用目标行缩进，保持可读性与嵌套配对。"""
    result = insert_instrumentation(_SOURCE, _point(begin_line=2, end_line=4))
    lines = result.splitlines()
    assert lines[1] == '    LV_PROFILER_BEGIN_TAG("draw_body");'
    assert lines[5] == '    LV_PROFILER_END_TAG("draw_body");'


def test_single_line_region_wraps_that_line():
    """begin==end 的单行区域被同一对埋点包裹。"""
    result = insert_instrumentation(_SOURCE, _point(begin_line=4, end_line=4))
    lines = result.splitlines()
    assert lines[3].strip() == 'LV_PROFILER_BEGIN_TAG("draw_body");'
    assert lines[4].strip() == "do_work(a, b);"
    assert lines[5].strip() == 'LV_PROFILER_END_TAG("draw_body");'


def test_render_helpers_emit_paired_macros():
    """render_begin_tag / render_end_tag 产出对应宏调用行。"""
    assert render_begin_tag("t") == 'LV_PROFILER_BEGIN_TAG("t");'
    assert render_end_tag("t") == 'LV_PROFILER_END_TAG("t");'
    assert render_begin_tag("t", indent="  ") == '  LV_PROFILER_BEGIN_TAG("t");'


def test_out_of_range_line_raises():
    """行号越界应报错，不静默产出错误源码。"""
    with pytest.raises(ValueError):
        insert_instrumentation(_SOURCE, _point(begin_line=10, end_line=10))


def test_invalid_span_rejected_at_construction():
    """end_line < begin_line 在构造点位时即被拒绝。"""
    with pytest.raises(ValueError):
        InstrumentationPoint(file="draw.c", tag="t", begin_line=4, end_line=2)


def test_empty_tag_rejected_at_construction():
    with pytest.raises(ValueError):
        InstrumentationPoint(file="draw.c", tag="   ", begin_line=1, end_line=1)


def test_insert_many_keeps_pairing_without_line_drift():
    """多个互不重叠区域一并插入（从后往前）时各自成对，且靠前区域行号不漂移。"""
    # 两个互不重叠区域：first=[2,2](int a=1;)，second=[4,4](do_work)。
    points = [
        _point(tag="first", begin_line=2, end_line=2),
        _point(tag="second", begin_line=4, end_line=4),
    ]
    result = insert_many(_SOURCE, points)
    lines = [ln.strip() for ln in result.splitlines()]

    # 每个标签恰各出现一次 BEGIN 与 END（配对未因行号漂移而错乱）
    for tag in ("first", "second"):
        assert lines.count(f'LV_PROFILER_BEGIN_TAG("{tag}");') == 1
        assert lines.count(f'LV_PROFILER_END_TAG("{tag}");') == 1

    # first 区域仍精确包住 int a = 1;（靠前区域行号未被靠后插入干扰）
    b_first = lines.index('LV_PROFILER_BEGIN_TAG("first");')
    assert lines[b_first + 1] == "int a = 1;"
    assert lines[b_first + 2] == 'LV_PROFILER_END_TAG("first");'

    # second 区域包住 do_work，且整体位于 first 之后（保序、不重叠）
    b_second = lines.index('LV_PROFILER_BEGIN_TAG("second");')
    assert lines[b_second + 1] == "do_work(a, b);"
    assert lines[b_second + 2] == 'LV_PROFILER_END_TAG("second");'
    assert lines.index('LV_PROFILER_END_TAG("first");') < b_second


# --------------------------------------------------------------------------- #
# 需求 6.2：标签作为 C 字符串字面量输出（转义）
# --------------------------------------------------------------------------- #


def test_tag_emitted_as_quoted_string_literal():
    """普通标签被双引号包裹为 C 字符串字面量，而非裸标识符（需求 6.2）。"""
    assert c_string_literal("draw_body") == '"draw_body"'


def test_string_literal_escapes_quotes_and_backslash():
    """双引号与反斜杠被正确转义，避免破坏源码。"""
    assert c_string_literal('a"b') == '"a\\"b"'
    assert c_string_literal("a\\b") == '"a\\\\b"'


def test_string_literal_escapes_control_chars():
    """换行/回车/制表符转义为可见转义序列。"""
    assert c_string_literal("a\nb") == '"a\\nb"'
    assert c_string_literal("a\rb") == '"a\\rb"'
    assert c_string_literal("a\tb") == '"a\\tb"'


def test_backslash_escaped_before_other_sequences():
    """先转义反斜杠，避免二次转义把字面反斜杠误当转义引导符。"""
    # 输入含一个反斜杠和一个双引号：应各自独立转义
    assert c_string_literal('\\"') == '"\\\\\\""'


def test_rendered_macros_use_escaped_literal():
    """渲染宏时标签同样以转义后的字符串字面量出现。"""
    point = _point(tag='weird"tag')
    result = insert_instrumentation(_SOURCE, point)
    assert 'LV_PROFILER_BEGIN_TAG("weird\\"tag");' in result
    assert 'LV_PROFILER_END_TAG("weird\\"tag");' in result


# --------------------------------------------------------------------------- #
# 需求 6.6：微秒 tick 精度报告分支
# --------------------------------------------------------------------------- #


def test_precision_insufficient_when_resolution_not_finer_than_subms():
    """tick 分辨率等于/粗于亚毫秒阈值 → 报告精度不足（需引入微秒级 tick）。"""
    # 恰等于阈值（1ms）：不优于 → 不足
    report = check_timing_precision(SUBMS_RESOLUTION_US)
    assert isinstance(report, PrecisionReport)
    assert report.sufficient is False
    assert report.tick_resolution_us == SUBMS_RESOLUTION_US
    assert report.required_resolution_us == SUBMS_RESOLUTION_US
    assert "微秒" in report.message

    # 更粗（10ms）：同样不足
    coarse = check_timing_precision(10_000)
    assert coarse.sufficient is False


def test_precision_sufficient_when_resolution_finer_than_subms():
    """tick 分辨率严格优于亚毫秒阈值 → 报告精度充分。"""
    report = check_timing_precision(100)  # 100us < 1000us
    assert report.sufficient is True
    assert report.tick_resolution_us == 100
    assert "可信" in report.message


def test_precision_respects_custom_required_resolution():
    """自定义所需分辨率参与判定：边界为严格小于。"""
    # 500us 相对所需 500us 不优于 → 不足
    assert check_timing_precision(500, required_resolution_us=500).sufficient is False
    # 499us 严格优于 500us → 充分
    assert check_timing_precision(499, required_resolution_us=500).sufficient is True


# --------------------------------------------------------------------------- #
# Instrumentor 类：编排纯编辑 + 注入的副作用/推理接口
# --------------------------------------------------------------------------- #


def _instrumentor(**overrides) -> Instrumentor:
    llm = FakeLLMProvider(responses=["decision"])
    device = overrides.pop("device", FakeDeviceIO())
    return Instrumentor(llm, device, **overrides)


def test_report_precision_uses_injected_tick_resolution():
    """report_precision 按注入的 tick 分辨率评估（需求 6.6）。"""
    coarse = _instrumentor(tick_resolution_us=1_000)
    assert coarse.report_precision().sufficient is False

    fine = _instrumentor(tick_resolution_us=50)
    assert fine.report_precision().sufficient is True


def test_edit_sources_returns_only_changed_files():
    """edit_sources 仅回传发生改动的文件，未涉及文件不回传。"""
    inst = _instrumentor()
    sources = {"draw.c": _SOURCE, "other.c": "int x;\n"}
    edited = inst.edit_sources(sources, [_point()])

    assert set(edited) == {"draw.c"}
    assert 'LV_PROFILER_BEGIN_TAG("draw_body");' in edited["draw.c"]


def test_edit_sources_rejects_point_referencing_missing_file():
    inst = _instrumentor()
    with pytest.raises(KeyError):
        inst.edit_sources({"draw.c": _SOURCE}, [_point(file="missing.c")])


def test_instrument_edits_and_reflashes_via_device():
    """instrument 编辑源码后经 DeviceIO 触发重编译与烧录（需求 6.3）。"""
    device = FakeDeviceIO()
    inst = _instrumentor(device=device)
    result = inst.instrument({"draw.c": _SOURCE}, [_point()])

    # 编辑产物含成对埋点
    assert 'LV_PROFILER_BEGIN_TAG("draw_body");' in result.edited_sources["draw.c"]
    # 触发了 build + flash（DeviceIO 副作用）
    assert isinstance(result.build, BuildResult) and result.build.success
    assert isinstance(result.flash, FlashResult) and result.flash.success
    assert result.reflashed is True
    assert ("build", None) in device.calls
    assert ("flash", None) in device.calls
    # 精度报告随结果返回
    assert isinstance(result.precision, PrecisionReport)


def test_instrument_dedupes_tags_preserving_order():
    """多点位的标签去重且保序，供历史记录使用。"""
    inst = _instrumentor()
    points = [
        _point(tag="a", begin_line=2, end_line=2),
        _point(tag="b", begin_line=3, end_line=3),
        _point(tag="a", begin_line=4, end_line=4),
    ]
    result = inst.instrument({"draw.c": _SOURCE}, points, reflash=False)
    assert result.tags == ["a", "b"]


def test_instrument_skips_flash_when_build_fails():
    """构建失败时不应烧录（需求 6.3 的守卫）。"""
    device = FakeDeviceIO(build_results=[BuildResult(success=False, error="boom")])
    inst = _instrumentor(device=device)
    result = inst.instrument({"draw.c": _SOURCE}, [_point()])

    assert isinstance(result.build, BuildResult) and result.build.success is False
    assert result.flash is None
    assert result.reflashed is False
    assert ("flash", None) not in device.calls


def test_instrument_no_reflash_skips_device_side_effects():
    """reflash=False 时不触发任何构建/烧录副作用，仅返回编辑产物。"""
    device = FakeDeviceIO()
    inst = _instrumentor(device=device)
    result = inst.instrument({"draw.c": _SOURCE}, [_point()], reflash=False)

    assert result.build is None
    assert result.flash is None
    assert result.reflashed is False
    assert ("build", None) not in device.calls


def test_instrument_write_callback_receives_edited_sources():
    """注入的 write 回调收到 file->新源码，用于隔离落盘副作用。"""
    inst = _instrumentor()
    written: dict[str, str] = {}
    inst.instrument(
        {"draw.c": _SOURCE},
        [_point()],
        write=lambda f, s: written.__setitem__(f, s),
        reflash=False,
    )
    assert set(written) == {"draw.c"}
    assert 'LV_PROFILER_END_TAG("draw_body");' in written["draw.c"]
