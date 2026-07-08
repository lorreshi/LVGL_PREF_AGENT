"""Frame_Analyzer：调用树 → 小体量慢帧/热点摘要（任务 8.5）。

作为**确定性普通工具**由 Orchestrator 调用（需求 13.2）。纯输入输出：给定相同的
``ParseResult`` 与配置，恒产生相同的 ``FrameReport``，不产生任何外部副作用
（Property 8）。

输入为 ``Trace_Parser`` 产出的按线程调用树（``ParseResult.trees_by_tid``）。
把**每一个线程内的根调用节点**视作一帧——在 LVGL 中，``lv_timer_handler`` 等顶层
调用每帧执行一次，其 ``duration_us`` 即该帧耗时。跨线程收集全部根节点后按
（begin_us, tid, func）稳定排序，得到确定的帧序（``SlowFrame.index``）。

本工具行为（严格对照需求 5）：

* **需求 5.1**：识别耗时**超过** ``cfg.frame_budget_us`` 的帧并标记为慢帧
  （``FrameReport.slow_frames``）。
* **需求 5.2**：按各函数在这些慢帧内的**聚合自耗时**（self-time，排除子节点耗时，
  避免父子重复计数）对函数排序，产出排名靠前的热点列表；列表裁剪至
  ``cfg.hotspot_top_n``（上下文预算机制 2）。
* **需求 5.4**：若没有任何帧超预算，则报告未检测到慢帧
  （``no_slow_frames=True``，``slow_frames`` / ``hotspots`` 为空）。

聚合统计（不逐帧进 context，Property 12）：``total_frames`` 为帧总数，
``p95_frame_us`` 为全部帧耗时的 P95（最近秩法）。

**摘要 token 预算**（design.md「摘要 token 预算」）：产出的 ``FrameReport`` 序列化
受 ``cfg.report_token_budget`` 约束；一旦超限即做**二次聚合**——先按半数递减截断
热点，再递减截断慢帧列表，直至序列化规模落入预算（函数名已在聚合时天然去重）。

概括字段 ``summary``（需求 5.3）由 LLM Analyzer 子智能体填写，本确定性工具留空
（``None``），以保持纯函数与可复现。
"""

from __future__ import annotations

import math

from embedded_device_agent.core.config.models import AppConfig
from embedded_device_agent.core.models import (
    ArtifactRef,
    CallTreeNode,
    FrameReport,
    HotspotEntry,
    ParseResult,
    SlowFrame,
)

__all__ = ["analyze_frames", "derive_frames"]


def _iter_nodes(node: CallTreeNode):
    """前序遍历某帧（根节点）子树，逐个产出调用节点。"""
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def derive_frames(tree: ParseResult) -> list[CallTreeNode]:
    """把按线程调用树归一为确定的帧序列（跨工具共享的单一真源）。

    每个线程内的**根调用节点**视作一帧；跨线程收集全部根节点后按
    （begin_us, tid, func）稳定排序，得到确定的帧序——其下标即
    ``SlowFrame.index``，亦是 ``query_trace`` 的 ``frame_index`` 坐标基准。
    """
    frames: list[CallTreeNode] = [
        root for roots in tree.trees_by_tid.values() for root in roots
    ]
    frames.sort(key=lambda n: (n.begin_us, n.tid, n.func))
    return frames


def _self_us(node: CallTreeNode) -> int:
    """节点自耗时 = 本节点耗时 - 直接子节点耗时之和（排除下层，避免重复计数）。

    依 Trace_Parser 不变量（子区间含于父区间，Property 3），该值恒 ``>= 0``。
    """
    child_us = sum(child.duration_us for child in node.children)
    return node.duration_us - child_us


def _frame_self_times(root: CallTreeNode) -> dict[str, list[int]]:
    """聚合单帧内各函数的自耗时与出现次数：``func -> [total_self_us, count]``。"""
    agg: dict[str, list[int]] = {}
    for node in _iter_nodes(root):
        rec = agg.setdefault(node.func, [0, 0])
        rec[0] += _self_us(node)
        rec[1] += 1
    return agg


def _dominant_func(root: CallTreeNode) -> str:
    """帧内自耗时最大的函数名；同值按函数名升序，保证确定性。"""
    agg = _frame_self_times(root)
    return min(agg.items(), key=lambda kv: (-kv[1][0], kv[0]))[0]


def _p95_us(durations: list[int]) -> int:
    """全部帧耗时的 P95（最近秩法：rank = ceil(0.95 * n)，1 基）。"""
    if not durations:
        return 0
    ordered = sorted(durations)
    rank = math.ceil(0.95 * len(ordered))
    idx = min(len(ordered) - 1, max(0, rank - 1))
    return ordered[idx]


def _estimate_tokens(report: FrameReport) -> int:
    """估算 ``FrameReport`` 序列化后的 token 数（约 4 字符/token 的通用启发式）。"""
    return math.ceil(len(report.model_dump_json()) / 4)


def _enforce_token_budget(report: FrameReport, budget: int) -> FrameReport:
    """二次聚合：超 ``report_token_budget`` 时按半数递减截断热点，再截断慢帧。

    优先截断热点（信息密度较低的尾部先被裁），仍超限再截断慢帧；两者各至少保留一项，
    避免摘要退化为空。``budget <= 0`` 视为不限制。
    """
    if budget <= 0:
        return report
    while _estimate_tokens(report) > budget:
        hotspots = report.hotspots
        slow_frames = report.slow_frames
        if len(hotspots) > 1:
            report = report.model_copy(
                update={"hotspots": hotspots[: max(1, len(hotspots) // 2)]}
            )
        elif len(slow_frames) > 1:
            report = report.model_copy(
                update={"slow_frames": slow_frames[: max(1, len(slow_frames) // 2)]}
            )
        else:
            break
    return report


def analyze_frames(
    tree: ParseResult,
    cfg: AppConfig,
    *,
    source: ArtifactRef | None = None,
    scenario: str = "",
    target_fps: int | None = None,
) -> FrameReport:
    """把调用树分析为小体量慢帧/热点摘要（需求 5.1 / 5.2 / 5.4）。

    Args:
        tree: ``Trace_Parser`` 产出的按线程调用树。
        cfg: 应用配置，提供 ``frame_budget_us`` / ``hotspot_top_n`` /
            ``report_token_budget``。
        source: 指向调用树 artifact 的引用，写入 ``FrameReport.source`` 供
            ``query_trace`` 下钻；缺省时置一最小占位引用（便于离线单测）。
        scenario: 采集场景标识，透传至报告。
        target_fps: 目标帧率；缺省时由 ``frame_budget_us`` 反推
            （``round(1e6 / frame_budget_us)``）。

    Returns:
        ``FrameReport``：仅含超预算慢帧、top-N 热点与聚合统计
        （``total_frames`` / ``p95_frame_us``），且序列化规模受
        ``report_token_budget`` 约束（超限二次聚合）。

    不变量：相同输入恒产生相同输出且无副作用（Property 8）；报告为小体量摘要，
    原始日志绝不进入其中（Property 12）。
    """
    budget_us = cfg.frame_budget_us
    if target_fps is None:
        target_fps = round(1_000_000 / budget_us) if budget_us > 0 else 0
    if source is None:
        # 离线单测的占位引用；实际运行由 Orchestrator 注入真实调用树 artifact。
        source = ArtifactRef(run_id="", kind="call_tree", path=".", size_bytes=0)

    # 每个线程的根节点即一帧；跨线程收集后按（起始, tid, 函数）稳定排序 → 确定帧序。
    frames = derive_frames(tree)

    total_frames = len(frames)
    p95_frame_us = _p95_us([f.duration_us for f in frames])

    # 需求 5.1：耗时超过帧预算的帧标记为慢帧。
    slow_indices = [
        i for i, f in enumerate(frames) if f.duration_us > budget_us
    ]

    # 需求 5.4：无慢帧分支——报告未检测到慢帧，热点/慢帧列表为空。
    if not slow_indices:
        report = FrameReport(
            scenario=scenario,
            target_fps=target_fps,
            frame_budget_us=budget_us,
            slow_frames=[],
            hotspots=[],
            total_frames=total_frames,
            p95_frame_us=p95_frame_us,
            source=source,
            summary=None,
            no_slow_frames=True,
        )
        return _enforce_token_budget(report, cfg.report_token_budget)

    slow_frames = [
        SlowFrame(
            index=i,
            duration_us=frames[i].duration_us,
            dominant_func=_dominant_func(frames[i]),
        )
        for i in slow_indices
    ]

    # 需求 5.2：按各函数在慢帧内的聚合自耗时排序热点。
    hot: dict[str, list[int]] = {}
    for i in slow_indices:
        for func, (self_us, count) in _frame_self_times(frames[i]).items():
            rec = hot.setdefault(func, [0, 0])
            rec[0] += self_us
            rec[1] += count

    ranked = sorted(hot.items(), key=lambda kv: (-kv[1][0], kv[0]))
    # 裁剪至 hotspot_top_n（上下文预算机制 2），并赋 1 基排名。
    hotspots = [
        HotspotEntry(func=func, total_us=total_us, call_count=count, rank=rank)
        for rank, (func, (total_us, count)) in enumerate(
            ranked[: cfg.hotspot_top_n], start=1
        )
    ]

    report = FrameReport(
        scenario=scenario,
        target_fps=target_fps,
        frame_budget_us=budget_us,
        slow_frames=slow_frames,
        hotspots=hotspots,
        total_frames=total_frames,
        p95_frame_us=p95_frame_us,
        source=source,
        summary=None,
        no_slow_frames=False,
    )
    return _enforce_token_budget(report, cfg.report_token_budget)
