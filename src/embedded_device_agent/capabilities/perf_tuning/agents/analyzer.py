"""Analyzer 子智能体：解读慢帧/热点，产出带证据的瓶颈概括（任务 11.4）。

严格对照 design.md「编排层与智能体层 → Analyzer」与需求 5.3 / 5.4：

* **确定性核心先行**：真正的慢帧识别与热点排序由确定性工具 ``Frame_Analyzer``
  （``analyze_frames``）完成，产出小体量的 ``FrameReport``（仅超预算慢帧、
  top-N 热点与聚合统计，Property 12）。Analyzer **不重算**这些事实，只**解读**它们。
* **仅推理步骤走 LLM**（design.md 核心原则 2）：Analyzer 把确定性事实喂给经
  ``BaseLLMProvider`` 注入的 chat model，产出一段自然语言的**瓶颈概括**，并要求
  其**引用具体的函数与帧**作为证据（需求 5.3）。LLM 藏在注入的 provider 之后，
  测试时可用 ``FakeLLMProvider`` 原样替换（Property 11）。
* **无慢帧分支**：当 ``FrameReport`` 无任何超预算帧（``no_slow_frames``）时，
  Analyzer **不调用 LLM**，直接产出「未检测到慢帧」的确定性报告（需求 5.4）。

产物写回 ``FrameReport.summary``（design.md 明确该字段由 Analyzer 填写，确定性
工具留空）。原始日志始终不进 context——Analyzer 只见到 ``FrameReport`` 摘要，
需要细节时由 Orchestrator 经 ``query_trace`` 按坐标下钻（Property 12）。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from embedded_device_agent.core.config.models import AppConfig
from embedded_device_agent.core.llm.base import BaseLLMProvider
from embedded_device_agent.core.models import (
    ArtifactRef,
    FrameReport,
    ParseResult,
)
from embedded_device_agent.capabilities.perf_tuning.tools.frame_analyzer import (
    analyze_frames,
)

__all__ = ["Analyzer"]


_SYSTEM_PROMPT = (
    "你是一名 LVGL 性能调优分析专家（Analyzer）。你会收到由确定性工具"
    "（Frame_Analyzer）算好的慢帧与热点证据。请基于这些证据，用简洁的中文"
    "概括最可能的性能瓶颈，并**明确引用具体的函数名与帧号**作为支撑证据。"
    "不要编造证据中不存在的函数或帧；不要索取或臆测原始日志内容。"
)


class Analyzer:
    """解读慢帧/热点、产出带证据瓶颈概括的 LLM 子智能体（需求 5.3 / 5.4）。

    副作用（LLM 调用）经注入的 :class:`BaseLLMProvider` 隔离，构造时不触达任何
    真实 LLM；仅在 :meth:`analyze` 处理**含慢帧**的报告时才惰性取用 chat model。
    """

    def __init__(self, llm: BaseLLMProvider) -> None:
        self._llm = llm

    # ---- 主入口：消费 Frame_Analyzer 的 FrameReport 并填写 summary ----
    def analyze(self, report: FrameReport) -> FrameReport:
        """解读 ``FrameReport``，返回填好 ``summary`` 的副本（不改动入参）。

        Args:
            report: ``Frame_Analyzer`` 产出的小体量摘要（慢帧 + top-N 热点 +
                聚合统计）。

        Returns:
            ``FrameReport`` 副本，``summary`` 已填入带证据的瓶颈概括；无慢帧时
            为确定性的「未检测到慢帧」报告（需求 5.4）。
        """
        # 需求 5.4：无慢帧分支——不调用 LLM，直接产出确定性报告。
        if report.no_slow_frames or not report.slow_frames:
            return report.model_copy(update={"summary": _no_slow_frames_summary(report)})

        # 需求 5.3：有热点/慢帧时，交由 LLM 概括瓶颈并引用具体函数与帧。
        summary = self._summarize_with_llm(report)
        if not summary:
            # LLM 返回空时回退到确定性证据概括，保证 summary 恒有证据支撑。
            summary = _fallback_summary(report)
        return report.model_copy(update={"summary": summary})

    # ---- 便捷入口：先跑确定性 Frame_Analyzer，再解读 ----
    def analyze_call_tree(
        self,
        tree: ParseResult,
        cfg: AppConfig,
        *,
        source: ArtifactRef | None = None,
        scenario: str = "",
        target_fps: int | None = None,
    ) -> FrameReport:
        """先调用确定性 ``analyze_frames`` 产出 ``FrameReport``，再解读并填 ``summary``。

        便于 Orchestrator / 测试直接从调用树一步得到「带概括」的报告；确定性事实
        与 LLM 概括的职责边界仍保持清晰（前者由 ``analyze_frames`` 负责）。
        """
        report = analyze_frames(
            tree,
            cfg,
            source=source,
            scenario=scenario,
            target_fps=target_fps,
        )
        return self.analyze(report)

    # ---- 内部：构造证据、调用 chat model、抽取文本 ----
    def _summarize_with_llm(self, report: FrameReport) -> str:
        """把确定性证据喂给注入的 chat model，返回其概括文本（已 strip）。"""
        model = self._llm.get_chat_model()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_build_evidence_prompt(report)),
        ]
        response = model.invoke(messages)
        return _content_to_text(getattr(response, "content", response)).strip()


def _no_slow_frames_summary(report: FrameReport) -> str:
    """无慢帧时的确定性报告文本（需求 5.4）。"""
    scenario = report.scenario or "所采集场景"
    return (
        f"在场景「{scenario}」中未检测到慢帧："
        f"全部 {report.total_frames} 帧均未超过帧预算 "
        f"{report.frame_budget_us}us（P95 帧耗时 {report.p95_frame_us}us）。"
    )


def _build_evidence_prompt(report: FrameReport) -> str:
    """把 ``FrameReport`` 中的慢帧与热点证据渲染为供 LLM 概括的输入文本。"""
    lines: list[str] = [
        f"场景：{report.scenario or '(未命名)'}",
        f"目标帧率：{report.target_fps} fps；帧预算：{report.frame_budget_us}us",
        f"总帧数：{report.total_frames}；P95 帧耗时：{report.p95_frame_us}us",
        f"超预算慢帧数：{len(report.slow_frames)}",
        "",
        "慢帧证据（帧号 / 帧耗时us / 主导函数）：",
    ]
    for sf in report.slow_frames:
        lines.append(
            f"  - 帧 #{sf.index}：{sf.duration_us}us，主导函数 {sf.dominant_func}"
        )
    lines.append("")
    lines.append("热点函数证据（排名 / 函数 / 聚合自耗时us / 调用次数）：")
    for hs in report.hotspots:
        lines.append(
            f"  - #{hs.rank} {hs.func}：{hs.total_us}us，共 {hs.call_count} 次"
        )
    lines.append("")
    lines.append(
        "请概括最可能的瓶颈，并引用上面具体的函数名与帧号作为证据。"
    )
    return "\n".join(lines)


def _fallback_summary(report: FrameReport) -> str:
    """LLM 返回空时的确定性兜底概括——直接引用最重证据（函数与帧）。"""
    parts: list[str] = []
    if report.hotspots:
        top = report.hotspots[0]
        parts.append(
            f"主要瓶颈疑似为函数 {top.func}（聚合自耗时 {top.total_us}us，"
            f"调用 {top.call_count} 次）"
        )
    if report.slow_frames:
        worst = max(report.slow_frames, key=lambda f: f.duration_us)
        parts.append(
            f"最慢帧为帧 #{worst.index}（{worst.duration_us}us，"
            f"主导函数 {worst.dominant_func}）"
        )
    if not parts:
        return _no_slow_frames_summary(report)
    return "；".join(parts) + "。"


def _content_to_text(content: object) -> str:
    """把 chat model 返回的 ``content`` 归一为纯文本。

    LangChain ``BaseChatModel`` 的响应 ``content`` 可能是字符串，也可能是
    内容块列表（如 ``[{"type": "text", "text": ...}]``）；此处统一抽取文本。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return str(content)
