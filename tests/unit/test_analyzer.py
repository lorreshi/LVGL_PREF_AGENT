"""任务 11.5：Analyzer 子智能体单元测试（需求 5.3 / 5.4）。

用确定性 ``FakeLLMProvider``（任务 16.1）注入 chat model，验证：

* **有慢帧分支（需求 5.3）**：Analyzer 调用（fake）LLM 概括瓶颈，``summary``
  被填入预置回放文本；同时其喂给 LLM 的证据 prompt 会引用具体的函数名与帧号
  （即「带证据」）。
* **无慢帧分支（需求 5.4）**：Analyzer **不调用 LLM**，直接产出确定性的
  「未检测到慢帧」概括；同样输入多次调用结果一致。

不触达真实硬件 / LLM / 网络：``FrameReport`` 输入直接由数据契约构造，LLM 经
``FakeLLMProvider`` 原样替换。
"""

from __future__ import annotations

from embedded_device_agent.capabilities.perf_tuning.agents.analyzer import (
    Analyzer,
    _build_evidence_prompt,
)
from embedded_device_agent.core.models import (
    ArtifactRef,
    FrameReport,
    HotspotEntry,
    SlowFrame,
)
from tests.harness import FakeLLMProvider

# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #


def _source() -> ArtifactRef:
    """指向调用树 artifact 的轻量引用（内容不进上下文，仅供下钻）。"""
    return ArtifactRef(
        run_id="run-0001",
        kind="call_tree",
        path="/tmp/call_tree.json",
        size_bytes=1234,
    )


def _report_with_slow_frames() -> FrameReport:
    """含慢帧 + 热点的报告：应触发 LLM 概括分支（需求 5.3）。"""
    return FrameReport(
        scenario="scroll",
        target_fps=60,
        frame_budget_us=16667,
        slow_frames=[
            SlowFrame(index=1, duration_us=30_000, dominant_func="lv_draw"),
            SlowFrame(index=3, duration_us=20_000, dominant_func="lv_refr"),
        ],
        hotspots=[
            HotspotEntry(func="lv_draw", total_us=40_000, call_count=2, rank=1),
            HotspotEntry(func="lv_refr", total_us=10_000, call_count=2, rank=2),
        ],
        total_frames=10,
        p95_frame_us=28_000,
        source=_source(),
    )


def _report_no_slow_frames() -> FrameReport:
    """无慢帧报告：应走确定性分支、不调用 LLM（需求 5.4）。"""
    return FrameReport(
        scenario="idle",
        target_fps=60,
        frame_budget_us=16667,
        slow_frames=[],
        hotspots=[],
        total_frames=8,
        p95_frame_us=12_000,
        no_slow_frames=True,
        source=_source(),
    )


# --------------------------------------------------------------------------- #
# 需求 5.3：有慢帧 → 调用（fake）LLM 概括，summary 取自预置回放文本
# --------------------------------------------------------------------------- #


def test_analyze_with_slow_frames_populates_summary_from_llm() -> None:
    """有慢帧时 Analyzer 调用 fake LLM，summary 等于预置回放文本。"""
    preset = "瓶颈疑似在 lv_draw（帧 #1 耗时 30000us）；lv_refr 亦偏高。"
    analyzer = Analyzer(FakeLLMProvider(responses=[preset]))

    report = _report_with_slow_frames()
    result = analyzer.analyze(report)

    # summary 来自预置 fake response（证明确实走了 LLM 概括分支）
    assert result.summary == preset
    # 不改动入参（返回副本）
    assert report.summary is None
    # 其余事实字段原样保留
    assert result.slow_frames == report.slow_frames
    assert result.hotspots == report.hotspots


def test_evidence_prompt_cites_specific_functions_and_frames() -> None:
    """喂给 LLM 的证据 prompt 引用了具体的函数名与帧号（带证据，需求 5.3）。"""
    report = _report_with_slow_frames()
    prompt = _build_evidence_prompt(report)

    # 引用具体帧号
    assert "帧 #1" in prompt
    assert "帧 #3" in prompt
    # 引用具体函数名（慢帧主导函数 + 热点函数）
    assert "lv_draw" in prompt
    assert "lv_refr" in prompt
    # 引用具体耗时证据
    assert "30000us" in prompt
    # 明确要求 LLM 引用函数与帧作为证据
    assert "证据" in prompt


def test_analyze_falls_back_to_deterministic_evidence_when_llm_empty() -> None:
    """LLM 返回空文本时回退到确定性证据概括，summary 恒引用具体证据。"""
    analyzer = Analyzer(FakeLLMProvider(responses=["   "]))

    result = analyzer.analyze(_report_with_slow_frames())

    assert result.summary  # 非空
    # 兜底概括直接引用最重证据（top 热点函数 + 最慢帧）
    assert "lv_draw" in result.summary
    assert "#1" in result.summary


# --------------------------------------------------------------------------- #
# 需求 5.4：无慢帧 → 不调用 LLM，确定性「未检测到慢帧」概括
# --------------------------------------------------------------------------- #


class _ExplodingLLMProvider(FakeLLMProvider):
    """若被调用即抛错的 provider——用于证明无慢帧分支不触达 LLM。"""

    def get_chat_model(self):  # type: ignore[override]
        raise AssertionError("无慢帧分支不应调用 LLM（需求 5.4）")


def test_no_slow_frames_branch_is_deterministic_without_llm() -> None:
    """无慢帧时产出确定性「未检测到慢帧」概括，且从不调用 LLM。"""
    analyzer = Analyzer(_ExplodingLLMProvider(responses=["unused"]))

    result = analyzer.analyze(_report_no_slow_frames())

    assert result.summary is not None
    assert "未检测到慢帧" in result.summary
    # 概括引用了确定性聚合事实
    assert "idle" in result.summary
    assert "8" in result.summary  # total_frames


def test_no_slow_frames_summary_is_stable_across_calls() -> None:
    """相同输入多次调用产出完全一致的确定性概括。"""
    analyzer = Analyzer(_ExplodingLLMProvider(responses=["unused"]))
    report = _report_no_slow_frames()

    first = analyzer.analyze(report).summary
    second = analyzer.analyze(report).summary
    assert first == second


def test_empty_slow_frames_without_flag_also_uses_deterministic_branch() -> None:
    """slow_frames 为空但未显式置 no_slow_frames 时，仍走确定性分支不调 LLM。"""
    report = _report_no_slow_frames().model_copy(update={"no_slow_frames": False})
    analyzer = Analyzer(_ExplodingLLMProvider(responses=["unused"]))

    result = analyzer.analyze(report)
    assert "未检测到慢帧" in (result.summary or "")
