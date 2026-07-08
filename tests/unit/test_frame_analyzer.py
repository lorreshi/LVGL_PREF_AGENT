"""任务 8.6：Frame_Analyzer（``analyze_frames`` / ``derive_frames``）单元 + 属性测试。

作为**确定性普通工具**（需求 13.2），本模块覆盖：

* 超预算慢帧判定（需求 5.1）；
* 按聚合自耗时（self-time）排序热点 + top-N 裁剪（需求 5.2）；
* 无慢帧分支 ``no_slow_frames=True``（需求 5.4）；
* 摘要 token 预算的二次聚合（design.md「摘要 token 预算」，需求 13.2）。

并以 hypothesis 覆盖两条属性：

* **Property 8：确定性核心无副作用**——相同输入恒产生相同 ``FrameReport``，
  且不修改输入。**Validates: Requirements 13.2**
* **Property 12：原始日志不进上下文**——``FrameReport`` 仅含 top-N 热点 +
  聚合统计，绝不承载原始调用树节点/日志。**Validates: Requirements 4.1, 5.2**

不触达真实硬件 / LLM / 网络：输入直接由 ``CallTreeNode`` 构造。
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from embedded_device_agent.capabilities.perf_tuning.tools.frame_analyzer import (
    analyze_frames,
    derive_frames,
)
from embedded_device_agent.core.config.models import (
    AppConfig,
    DeviceConfig,
    LLMConfig,
    RetrieverConfig,
)
from embedded_device_agent.core.models import CallTreeNode, ParseResult

# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #


def _node(
    func: str,
    begin: int,
    duration: int,
    *,
    tid: int = 1,
    children: list[CallTreeNode] | None = None,
) -> CallTreeNode:
    """构造一个调用节点，``end_us`` 由 ``begin + duration`` 计算。"""
    return CallTreeNode(
        func=func,
        tid=tid,
        begin_us=begin,
        end_us=begin + duration,
        duration_us=duration,
        children=children or [],
    )


def _parse_result(roots: list[CallTreeNode]) -> ParseResult:
    """按 tid 分组根节点，组成 ``ParseResult``（每个根即一帧）。"""
    trees: dict[int, list[CallTreeNode]] = {}
    for r in roots:
        trees.setdefault(r.tid, []).append(r)
    return ParseResult(trees_by_tid=trees, unmatched=[])


def _make_config(**overrides) -> AppConfig:
    base = dict(
        llm=LLMConfig(type="anthropic", model="claude", api_key_env="ANTHROPIC_API_KEY"),
        device=DeviceConfig(type="fake"),
        retriever=RetrieverConfig(),
        frame_budget_us=16667,
        hotspot_top_n=20,
        report_token_budget=0,  # 单测默认不限制，聚焦分析逻辑
        mode="full_auto",
        max_iterations=5,
    )
    base.update(overrides)
    return AppConfig(**base)


# --------------------------------------------------------------------------- #
# 需求 5.1：超预算慢帧判定
# --------------------------------------------------------------------------- #


def test_over_budget_slow_frame_detection():
    """耗时严格大于帧预算的帧被标记为慢帧，边界（恰等于预算）不算慢帧。"""
    cfg = _make_config(frame_budget_us=16667)
    frames = [
        _node("lv_timer_handler", begin=0, duration=10_000),  # 未超
        _node("lv_timer_handler", begin=100_000, duration=30_000),  # 超
        _node("lv_timer_handler", begin=200_000, duration=16_667),  # 恰等于，不超
        _node("lv_timer_handler", begin=300_000, duration=20_000),  # 超
    ]
    report = analyze_frames(_parse_result(frames), cfg, scenario="scroll")

    assert report.no_slow_frames is False
    assert [sf.index for sf in report.slow_frames] == [1, 3]
    assert [sf.duration_us for sf in report.slow_frames] == [30_000, 20_000]
    assert report.total_frames == 4
    assert report.scenario == "scroll"
    assert report.frame_budget_us == 16667
    # target_fps 由帧预算反推
    assert report.target_fps == round(1_000_000 / 16667)


def test_target_fps_override_is_passed_through():
    cfg = _make_config()
    frames = [_node("lv_timer_handler", begin=0, duration=30_000)]
    report = analyze_frames(_parse_result(frames), cfg, target_fps=90)
    assert report.target_fps == 90


# --------------------------------------------------------------------------- #
# 需求 5.2：按聚合自耗时排序热点 + top-N 裁剪
# --------------------------------------------------------------------------- #


def _slow_frame_with_hotspots(begin: int) -> CallTreeNode:
    """一个超预算帧：root 自耗时 5000，lv_draw 20000，lv_refr 5000。"""
    return _node(
        "lv_timer_handler",
        begin=begin,
        duration=30_000,
        children=[
            _node("lv_draw", begin=begin, duration=20_000),
            _node("lv_refr", begin=begin + 20_000, duration=5_000),
        ],
    )


def test_hotspot_ranking_by_aggregate_self_time():
    """热点按聚合自耗时降序；同值按函数名升序，保证确定性。"""
    cfg = _make_config(hotspot_top_n=20)
    report = analyze_frames(_parse_result([_slow_frame_with_hotspots(0)]), cfg)

    # self: lv_draw=20000, lv_refr=5000, lv_timer_handler=5000（30000-25000）
    # 排序：lv_draw > (lv_refr == lv_timer_handler，按名升序 lv_refr 先)
    assert [h.func for h in report.hotspots] == ["lv_draw", "lv_refr", "lv_timer_handler"]
    assert [h.total_us for h in report.hotspots] == [20_000, 5_000, 5_000]
    assert [h.rank for h in report.hotspots] == [1, 2, 3]


def test_hotspot_self_time_aggregates_across_slow_frames():
    """跨多个慢帧的同名函数自耗时应累加，调用次数亦累加。"""
    cfg = _make_config()
    frames = [_slow_frame_with_hotspots(0), _slow_frame_with_hotspots(100_000)]
    report = analyze_frames(_parse_result(frames), cfg)

    hs = {h.func: h for h in report.hotspots}
    assert hs["lv_draw"].total_us == 40_000  # 20000 * 2
    assert hs["lv_draw"].call_count == 2
    assert hs["lv_timer_handler"].total_us == 10_000  # 5000 * 2


def test_hotspot_top_n_trimming():
    """热点列表裁剪至 ``hotspot_top_n``（上下文预算机制 2）。"""
    cfg = _make_config(hotspot_top_n=2)
    report = analyze_frames(_parse_result([_slow_frame_with_hotspots(0)]), cfg)

    assert len(report.hotspots) == 2  # 3 个函数被裁剪至 top-2
    assert [h.func for h in report.hotspots] == ["lv_draw", "lv_refr"]


def test_slow_frame_dominant_func_is_max_self_time():
    """慢帧的 dominant_func 为帧内自耗时最大的函数。"""
    cfg = _make_config()
    report = analyze_frames(_parse_result([_slow_frame_with_hotspots(0)]), cfg)
    assert report.slow_frames[0].dominant_func == "lv_draw"


# --------------------------------------------------------------------------- #
# 需求 5.4：无慢帧分支
# --------------------------------------------------------------------------- #


def test_no_slow_frames_branch():
    """无帧超预算时报告 no_slow_frames=True，慢帧/热点均为空。"""
    cfg = _make_config(frame_budget_us=16667)
    frames = [
        _node("lv_timer_handler", begin=0, duration=10_000),
        _node("lv_timer_handler", begin=100_000, duration=16_667),  # 恰等于预算
    ]
    report = analyze_frames(_parse_result(frames), cfg)

    assert report.no_slow_frames is True
    assert report.slow_frames == []
    assert report.hotspots == []
    # 聚合统计仍然计算
    assert report.total_frames == 2
    assert report.p95_frame_us == 16_667


def test_empty_input_is_no_slow_frames():
    cfg = _make_config()
    report = analyze_frames(_parse_result([]), cfg)
    assert report.no_slow_frames is True
    assert report.total_frames == 0
    assert report.p95_frame_us == 0


# --------------------------------------------------------------------------- #
# 需求 13.2 / design：token 预算二次聚合
# --------------------------------------------------------------------------- #


def _many_slow_frames(count: int) -> list[CallTreeNode]:
    """构造 ``count`` 个超预算帧，各含若干不同函数的热点。"""
    frames: list[CallTreeNode] = []
    for i in range(count):
        begin = i * 100_000
        children = [
            _node(f"func_{i}_{j}", begin=begin + j * 1_000, duration=2_000)
            for j in range(6)
        ]
        frames.append(
            _node("lv_timer_handler", begin=begin, duration=30_000, children=children)
        )
    return frames


def test_token_budget_secondary_aggregation_trims_report():
    """设置很小的 report_token_budget 时，热点/慢帧被二次聚合裁剪。"""
    frames = _parse_result(_many_slow_frames(8))

    unlimited = analyze_frames(frames, _make_config(report_token_budget=0))
    limited = analyze_frames(frames, _make_config(report_token_budget=150))

    # 二次聚合后规模严格变小（先裁热点，再裁慢帧）
    assert len(limited.hotspots) < len(unlimited.hotspots)
    assert len(limited.hotspots) >= 1  # 至少保留一项，避免退化为空
    assert len(limited.slow_frames) <= len(unlimited.slow_frames)
    # 裁剪后序列化规模确实落入预算（约 4 字符/token）
    assert len(limited.model_dump_json()) / 4 <= 150


def test_token_budget_zero_means_unlimited():
    """budget<=0 视为不限制，报告不被裁剪。"""
    frames = _parse_result(_many_slow_frames(4))
    report = analyze_frames(frames, _make_config(report_token_budget=0))
    # 4 帧 * 6 子函数 + root = 25 个不同函数，未超 top_n(20) 则应有 20 个
    assert len(report.hotspots) == 20
    assert len(report.slow_frames) == 4


# --------------------------------------------------------------------------- #
# derive_frames：确定帧序
# --------------------------------------------------------------------------- #


def test_derive_frames_stable_order_across_threads():
    """跨线程根节点按 (begin_us, tid, func) 稳定排序得到确定帧序。"""
    roots = [
        _node("b", begin=100, duration=10, tid=2),
        _node("a", begin=100, duration=10, tid=1),
        _node("z", begin=0, duration=10, tid=1),
    ]
    frames = derive_frames(_parse_result(roots))
    assert [(f.begin_us, f.tid, f.func) for f in frames] == [
        (0, 1, "z"),
        (100, 1, "a"),
        (100, 2, "b"),
    ]


# --------------------------------------------------------------------------- #
# hypothesis 生成器（约束到调用树的合理输入空间）
# --------------------------------------------------------------------------- #

_func_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=16,
)


@st.composite
def _frame_roots(draw: st.DrawFn) -> CallTreeNode:
    """一个帧（根节点）：子节点耗时之和不超过父耗时（尊重 Trace_Parser 不变量）。"""
    begin = draw(st.integers(min_value=0, max_value=1_000_000))
    duration = draw(st.integers(min_value=1, max_value=100_000))
    n_children = draw(st.integers(min_value=0, max_value=4))
    children: list[CallTreeNode] = []
    cursor = begin
    remaining = duration
    for _ in range(n_children):
        if remaining <= 1:
            break
        child_dur = draw(st.integers(min_value=1, max_value=remaining))
        children.append(_node(draw(_func_names), begin=cursor, duration=child_dur))
        cursor += child_dur
        remaining -= child_dur
    return _node(draw(_func_names), begin=begin, duration=duration, children=children)


@st.composite
def _parse_results(draw: st.DrawFn) -> ParseResult:
    roots = draw(st.lists(_frame_roots(), max_size=12))
    # 分散到 1~3 个线程，得到跨线程帧序
    trees: dict[int, list[CallTreeNode]] = {}
    for idx, r in enumerate(roots):
        tid = (idx % 3) + 1
        node = r.model_copy(update={"tid": tid})
        trees.setdefault(tid, []).append(node)
    return ParseResult(trees_by_tid=trees, unmatched=[])


_configs = st.builds(
    _make_config,
    frame_budget_us=st.integers(min_value=1, max_value=50_000),
    hotspot_top_n=st.integers(min_value=1, max_value=10),
    report_token_budget=st.sampled_from([0, 200, 4000]),
)


def _all_funcs(tree: ParseResult) -> set[str]:
    funcs: set[str] = set()

    def _walk(n: CallTreeNode) -> None:
        funcs.add(n.func)
        for c in n.children:
            _walk(c)

    for roots in tree.trees_by_tid.values():
        for root in roots:
            _walk(root)
    return funcs


# --------------------------------------------------------------------------- #
# Property 8：确定性核心无副作用
# **Validates: Requirements 13.2**
# --------------------------------------------------------------------------- #


@settings(max_examples=150, deadline=None)
@given(tree=_parse_results(), cfg=_configs)
def test_property8_deterministic_no_side_effects(tree: ParseResult, cfg: AppConfig):
    """相同输入恒产生相同 FrameReport，且不修改输入（无副作用）。"""
    snapshot = tree.model_copy(deep=True)
    cfg_snapshot = cfg.model_copy(deep=True)

    first = analyze_frames(tree, cfg, scenario="s", target_fps=60)
    second = analyze_frames(tree, cfg, scenario="s", target_fps=60)

    # 确定性：两次调用结果逐字节一致
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    # 无副作用：输入未被修改
    assert tree == snapshot
    assert cfg == cfg_snapshot


# --------------------------------------------------------------------------- #
# Property 12：原始日志不进上下文（仅 top-N + 聚合统计进 FrameReport）
# **Validates: Requirements 4.1, 5.2**
# --------------------------------------------------------------------------- #

_FRAME_REPORT_FIELDS = {
    "scenario",
    "target_fps",
    "frame_budget_us",
    "slow_frames",
    "hotspots",
    "total_frames",
    "p95_frame_us",
    "source",
    "summary",
    "no_slow_frames",
}


@settings(max_examples=150, deadline=None)
@given(tree=_parse_results(), cfg=_configs)
def test_property12_report_is_bounded_summary(tree: ParseResult, cfg: AppConfig):
    """FrameReport 仅含 top-N 热点 + 聚合统计，绝不承载原始调用树节点。"""
    report = analyze_frames(tree, cfg)

    # 结构上仅暴露既定的小体量字段，不含任何原始节点/日志列表
    assert set(report.model_dump().keys()) == _FRAME_REPORT_FIELDS

    # 热点裁剪至 top-N（上下文预算机制 2）
    assert len(report.hotspots) <= cfg.hotspot_top_n

    # 报告内出现的函数名必来自输入调用树（非凭空捏造的原始内容）
    tree_funcs = _all_funcs(tree)
    for h in report.hotspots:
        assert h.func in tree_funcs
    for sf in report.slow_frames:
        assert sf.dominant_func in tree_funcs

    # 聚合统计始终存在且与帧总数一致
    assert report.total_frames == len(derive_frames(tree))
    # summary 由 LLM Analyzer 填写，确定性工具留空
    assert report.summary is None

    # 若设了 token 预算，序列化规模落入其中（约 4 字符/token）
    if cfg.report_token_budget > 0:
        assert len(report.model_dump_json()) / 4 <= cfg.report_token_budget
