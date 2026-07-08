"""任务 8.2：Trace_Filter 单元测试 + 属性测试。

覆盖 `trace_filter`（``src/embedded_device_agent/capabilities/perf_tuning/tools/
trace_filter.py``）在真实磁盘工件上的行为：

单元测试
* 交错破坏 begin/end 配对时的**排除与计数**、**受损区间报告**（需求 3.2/3.3）。
* 已清洗工件中被保留事件的**时间戳保留**（微秒精度，需求 3.4）。

属性测试（hypothesis）
* **Property 2：B/E 配对守恒** —— ``retained + excluded`` 等于原始日志解析出的
  事件总数；被保留的事件必然成对（同 tid、正确嵌套、E 时间戳不早于 B）。
  **Validates: Requirements 3.2, 3.3**
* **Property 8：确定性核心无副作用** —— 相同输入内容恒产生相同的输出与计数，
  且除写出已清洗工件外不产生任何外部副作用（不改动原始日志、不创建其他文件）。
  **Validates: Requirements 13.2**
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from embedded_device_agent.capabilities.perf_tuning.tools.trace_filter import trace_filter
from embedded_device_agent.core.models import RawTraceArtifact

# --------------------------------------------------------------------------- #
# 辅助：构造 RawTraceArtifact、写原始日志、解析已清洗 systrace 工件
# --------------------------------------------------------------------------- #


def _artifact(path: Path) -> RawTraceArtifact:
    """把磁盘上的原始日志文件包装为 RawTraceArtifact（其余字段取合理占位值）。"""
    return RawTraceArtifact(
        run_id="run-test",
        path=path,
        captured_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        duration_s=1.0,
        baud=115200,
    )


def _write_raw(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _event_line(kind: str, tid: int, func: str, ts_us: int) -> str:
    """按 LVGL profiler systrace 格式构造一条事件行（时间戳为秒.微秒）。"""
    sec, frac = divmod(ts_us, 1_000_000)
    return f"LVGL-{tid} [0] {sec}.{frac:06d}: tracing_mark_write: {kind}|{tid}|{func}"


def _parse_systrace(path: Path) -> list[tuple[str, int, str, int]]:
    """把已清洗 systrace 工件解析为 (kind, tid, func, ts_us) 列表（跳过注释）。"""
    import re

    pat = re.compile(
        r"^\s*\S+-\d+\s+\[\d+\]\s+(?P<sec>\d+)\.(?P<frac>\d+)\s*:\s+"
        r"tracing_mark_write\s*:\s+(?P<kind>[BE])\|(?P<tid>\d+)\|(?P<func>.+?)\s*$"
    )
    events: list[tuple[str, int, str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = pat.match(line)
        assert m is not None, f"已清洗工件出现无法解析的行: {line!r}"
        ts_us = int(m.group("sec")) * 1_000_000 + int((m.group("frac") + "000000")[:6])
        events.append((m.group("kind"), int(m.group("tid")), m.group("func"), ts_us))
    return events


# --------------------------------------------------------------------------- #
# 单元测试：交错破坏配对 → 排除、计数、受损区间（需求 3.2 / 3.3）
# --------------------------------------------------------------------------- #


def test_interleaving_breaks_pairing_excludes_and_counts(tmp_path: Path):
    """交错噪声与孤立事件被排除，成对事件被保留，计数与受损区间正确。"""
    raw = tmp_path / "raw.log"
    _write_raw(
        raw,
        [
            "# tracer: nop",  # 行1：注释，跳过（不计事件也不计损坏）
            _event_line("B", 1, "funcA", 100_000_000),  # 行2：成对保留
            _event_line("E", 1, "funcA", 100_000_500),  # 行3：成对保留
            _event_line("B", 1, "funcB", 100_001_000),  # 行4：无匹配 E → 排除
            "!!! interleaved garbage from other thread !!!",  # 行5：无法解析 → 损坏
            _event_line("E", 1, "funcC", 100_002_000),  # 行6：孤立 E → 排除
        ],
    )
    clean = tmp_path / "clean.systrace"

    result = trace_filter(_artifact(raw), clean_path=clean)

    # 解析出的事件总数 = 4（funcA 的 B/E、funcB 的 B、funcC 的 E）。
    assert result.retained == 2
    assert result.excluded == 2
    assert result.retained + result.excluded == 4
    # 受损区间 = 无法解析行(5) + 被排除事件行(4, 6) 合并 → (4, 6)。
    assert result.corrupted_regions == [(4, 6)]

    # 已清洗工件仅含成对保留的 funcA 事件。
    events = _parse_systrace(result.clean_path)
    assert events == [
        ("B", 1, "funcA", 100_000_000),
        ("E", 1, "funcA", 100_000_500),
    ]


def test_nested_interleaving_retains_outer_pair_excludes_inner(tmp_path: Path):
    """交错丢失内层 E 时，保留完好的外层配对，仅排除被破坏的内层 B。"""
    raw = tmp_path / "raw.log"
    _write_raw(
        raw,
        [
            _event_line("B", 7, "outer", 10_000),  # 行1
            _event_line("B", 7, "inner", 20_000),  # 行2：内层，无匹配 E → 排除
            _event_line("E", 7, "outer", 30_000),  # 行3：与外层 B 配对保留
        ],
    )
    clean = tmp_path / "clean.systrace"

    result = trace_filter(_artifact(raw), clean_path=clean)

    assert result.retained == 2  # outer B + outer E
    assert result.excluded == 1  # inner B
    assert result.retained + result.excluded == 3
    assert result.corrupted_regions == [(2, 2)]  # 内层 B 所在行

    events = _parse_systrace(result.clean_path)
    assert events == [
        ("B", 7, "outer", 10_000),
        ("E", 7, "outer", 30_000),
    ]


# --------------------------------------------------------------------------- #
# 单元测试：时间戳保留（微秒精度，需求 3.4）
# --------------------------------------------------------------------------- #


def test_retained_event_timestamps_preserved(tmp_path: Path):
    """被保留事件在已清洗工件中无损保留其时间戳（含亚毫秒精度与补零）。"""
    raw = tmp_path / "raw.log"
    # 12.5 → 12_500_000us（小数补零到 6 位）；2892.002993 → 2892_002_993us（全精度保留）。
    _write_raw(
        raw,
        [
            "LVGL-2 [0] 12.5: tracing_mark_write: B|2|render",
            "LVGL-2 [0] 2892.002993: tracing_mark_write: E|2|render",
        ],
    )
    clean = tmp_path / "clean.systrace"

    result = trace_filter(_artifact(raw), clean_path=clean)

    assert result.retained == 2
    assert result.excluded == 0

    text = result.clean_path.read_text(encoding="utf-8")
    # 时间戳无损：12.5 归一化为 12.500000，2892.002993 原样保留（微秒精度）。
    assert "12.500000: tracing_mark_write: B|2|render" in text
    assert "2892.002993: tracing_mark_write: E|2|render" in text

    events = _parse_systrace(result.clean_path)
    assert events == [
        ("B", 2, "render", 12_500_000),
        ("E", 2, "render", 2_892_002_993),
    ]


# --------------------------------------------------------------------------- #
# 属性测试生成器：合成带有交错噪声的原始 profiler 日志
# --------------------------------------------------------------------------- #

_FUNCS = ["a", "b", "c", "lv_draw_rect", "绘制"]  # 含一个 Unicode 函数名


@st.composite
def _raw_logs(draw: st.DrawFn) -> tuple[str, int]:
    """生成 (原始日志文本, 可解析事件总数)。

    随机交错 B / E 事件（跨多个 tid、少量函数名）与无法解析的噪声行；时间戳按
    发射顺序严格递增（贴合真实 profiler 的时序），使成对事件天然满足「E 不早于 B」。
    噪声行保证不匹配事件正则、不以 ``#`` 开头、非空 → 计为受损而非事件。
    """
    n = draw(st.integers(min_value=0, max_value=30))
    ts = draw(st.integers(min_value=0, max_value=1_000))
    lines: list[str] = []
    num_events = 0
    for _ in range(n):
        ts += draw(st.integers(min_value=1, max_value=5_000))  # 严格递增
        kind = draw(st.sampled_from(["B", "E", "noise"]))
        if kind == "noise":
            token = draw(
                st.text(
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd"),
                        whitelist_characters="_- ",
                    ),
                    min_size=1,
                    max_size=20,
                )
            )
            lines.append("noise-" + token)  # 不以 # 开头、不匹配事件正则
        else:
            tid = draw(st.integers(min_value=1, max_value=3))
            func = draw(st.sampled_from(_FUNCS))
            lines.append(_event_line(kind, tid, func, ts))
            num_events += 1
    return "\n".join(lines) + ("\n" if lines else ""), num_events


def _assert_properly_paired(events: list[tuple[str, int, str, int]]) -> None:
    """断言事件序列按 tid 严格嵌套配对：每个 E 关闭同 tid、同名、最近的未闭合 B。"""
    stacks: dict[int, list[tuple[str, int]]] = {}  # tid -> [(func, ts_us)]
    for kind, tid, func, ts_us in events:
        stack = stacks.setdefault(tid, [])
        if kind == "B":
            stack.append((func, ts_us))
        else:  # E：保留事件必成对 → 栈非空、栈顶同名、E 时间戳不早于 B
            assert stack, "已保留事件出现无匹配 B 的孤立 E"
            top_func, top_ts = stack.pop()
            assert top_func == func, "已保留事件嵌套不正确（E 未关闭最近的同名 B）"
            assert ts_us >= top_ts, "已保留事件 E 的时间戳早于其配对 B"
    for tid, stack in stacks.items():
        assert not stack, f"tid={tid} 存在未闭合的 B（已保留事件未成对）"


# --------------------------------------------------------------------------- #
# Property 2：B/E 配对守恒
# **Validates: Requirements 3.2, 3.3**
# --------------------------------------------------------------------------- #


@settings(max_examples=200, deadline=None)
@given(payload=_raw_logs())
def test_property2_be_pairing_conservation(payload: tuple[str, int]):
    """retained + excluded == 解析事件总数；被保留事件必然成对。"""
    raw_text, num_events = payload
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        raw = base / "raw.log"
        raw.write_text(raw_text, encoding="utf-8")
        clean = base / "clean.systrace"

        result = trace_filter(_artifact(raw), clean_path=clean)

        # 守恒：保留 + 排除 == 从原始日志解析出的事件总数。
        assert result.retained + result.excluded == num_events
        assert result.retained >= 0 and result.excluded >= 0

        # 被保留事件必然成对（同 tid、正确嵌套、E 不早于 B）。
        retained_events = _parse_systrace(result.clean_path)
        assert len(retained_events) == result.retained
        _assert_properly_paired(retained_events)


# --------------------------------------------------------------------------- #
# Property 8：确定性核心无副作用
# **Validates: Requirements 13.2**
# --------------------------------------------------------------------------- #


@settings(max_examples=150, deadline=None)
@given(payload=_raw_logs())
def test_property8_deterministic_no_side_effects(payload: tuple[str, int]):
    """相同输入恒产生相同输出/计数；除写出已清洗工件外无外部副作用。"""
    raw_text, _ = payload

    # 两次独立运行使用相同输入内容，位于隔离的临时目录中。
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        base1, base2 = Path(td1), Path(td2)
        raw1, raw2 = base1 / "raw.log", base2 / "raw.log"
        raw1.write_text(raw_text, encoding="utf-8")
        raw2.write_text(raw_text, encoding="utf-8")
        clean1, clean2 = base1 / "clean.systrace", base2 / "clean.systrace"

        r1 = trace_filter(_artifact(raw1), clean_path=clean1)
        r2 = trace_filter(_artifact(raw2), clean_path=clean2)

        # 确定性：计数与受损区间逐字段相等。
        assert (r1.retained, r1.excluded) == (r2.retained, r2.excluded)
        assert r1.corrupted_regions == r2.corrupted_regions
        # 确定性：已清洗工件内容字节级一致。
        assert clean1.read_bytes() == clean2.read_bytes()

        # 无副作用：原始日志内容未被改动。
        assert raw1.read_text(encoding="utf-8") == raw_text
        # 无副作用：目录中仅存在原始日志与已清洗工件两个文件。
        assert {p.name for p in base1.iterdir()} == {"raw.log", "clean.systrace"}


def test_property8_repeated_calls_same_path_identical(tmp_path: Path):
    """对同一输出路径重复调用产生字节级一致的已清洗工件（确定性/幂等）。"""
    raw = tmp_path / "raw.log"
    _write_raw(
        raw,
        [
            _event_line("B", 1, "f", 1_000),
            _event_line("B", 2, "g", 2_000),
            _event_line("E", 1, "f", 3_000),
            "garbage",
            _event_line("E", 2, "g", 4_000),
        ],
    )
    clean = tmp_path / "clean.systrace"

    r1 = trace_filter(_artifact(raw), clean_path=clean)
    first = clean.read_bytes()
    r2 = trace_filter(_artifact(raw), clean_path=clean)
    second = clean.read_bytes()

    assert first == second
    assert (r1.retained, r1.excluded, r1.corrupted_regions) == (
        r2.retained,
        r2.excluded,
        r2.corrupted_regions,
    )
