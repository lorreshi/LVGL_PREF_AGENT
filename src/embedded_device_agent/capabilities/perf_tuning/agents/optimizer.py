"""Optimizer 子智能体：针对叶子热点提出并（按模式）应用源码级优化（任务 11.8）。

严格对照 design.md「编排层与智能体层 → Orchestrator」子智能体表与需求 7：

* **需求 7.1**：当一个叶子级热点被隔离出来时，提出一项或多项源码级优化，
  每一项都**引用目标函数**（``target_func``）以及**预期的性能效果**
  （``expected_effect``）。该推理步骤由注入的 :class:`BaseLLMProvider` 驱动。
* **需求 7.2**：在 ``full_auto`` 模式下，将所选优化应用到 C 源码，并经注入的
  :class:`DeviceIO` 重新编译（``build``）与烧录（``flash``）。
* **需求 7.3**：在 ``half_auto`` 模式下，仅**呈现**所提出的优化待开发者批准，
  在获批前**不应用**任何改动（返回 ``pending_approval`` 状态 + 干预请求）。
* **需求 7.4**：当一项优化被应用时，返回所应用改动的摘要及其目标函数，供
  Orchestrator 记录到运行状态（``TuningRunState.optimizations_applied``）。

设计原则「一切副作用皆经接口注入」在此落地：LLM 提议经 ``BaseLLMProvider``、
源码改动经可注入的 :class:`SourceEditor`、编译烧录经 ``DeviceIO``——三者在测试中
均可替换为 fake，使本子智能体可脱离真实硬件与真实 LLM 离线自测。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from embedded_device_agent.core.device.base import DeviceIO
from embedded_device_agent.core.device.models import (
    BuildResult,
    FlashResult,
    InterventionRequest,
)
from embedded_device_agent.core.llm.base import BaseLLMProvider
from embedded_device_agent.core.models import FrameReport, HotspotEntry

__all__ = [
    "OptimizationProposal",
    "OptimizationProposalList",
    "AppliedOptimization",
    "OptimizerResult",
    "SourceEditor",
    "FileSourceEditor",
    "Optimizer",
]


# ---- 数据契约（可独立构造、可无损 JSON 序列化，支撑测试与 API 边界）----
class OptimizationProposal(BaseModel):
    """一项针对叶子热点的源码级优化提议（需求 7.1）。

    每项提议都必须**引用目标函数**（``target_func``）与**预期性能效果**
    （``expected_effect``），并给出可应用的最小代码改动（``file_path`` +
    ``original_snippet`` → ``optimized_snippet``），使 ``full_auto`` 模式可直接落地、
    ``half_auto`` 模式可清晰呈现待批准。
    """

    target_func: str  # 引用的目标函数（需求 7.1）
    title: str  # 优化的简短标题
    description: str  # 优化做法说明
    expected_effect: str  # 预期性能效果（需求 7.1）
    file_path: str  # 目标 C 源文件路径
    original_snippet: str  # 待替换的原始代码片段
    optimized_snippet: str  # 优化后的代码片段
    rationale: str = ""  # 与热点证据的关联依据


class OptimizationProposalList(BaseModel):
    """LLM 结构化输出容器：一次可返回多项提议（需求 7.1「一项或多项」）。"""

    proposals: list[OptimizationProposal] = Field(default_factory=list)


class AppliedOptimization(BaseModel):
    """一项已应用优化的记录，供 Orchestrator 写入运行状态（需求 7.4）。"""

    target_func: str  # 目标函数（需求 7.4）
    file_path: str
    summary: str  # 所应用改动的摘要（需求 7.4）
    build: BuildResult | None = None
    flash: FlashResult | None = None
    # 当 build/flash 无法脚本化（半自动后端）时降级为人在环干预请求（需求 14.2）
    build_intervention: InterventionRequest | None = None
    flash_intervention: InterventionRequest | None = None


class OptimizerResult(BaseModel):
    """Optimizer 一次运行的结构化结果。

    * ``no_hotspot``：未提供叶子热点，无可优化对象。
    * ``pending_approval``：``half_auto`` 模式已呈现提议、等待开发者批准（未应用）。
    * ``applied``：``full_auto`` 模式已选定并应用一项优化（含重编译/烧录）。
    """

    status: Literal["no_hotspot", "pending_approval", "applied"]
    proposals: list[OptimizationProposal] = Field(default_factory=list)
    selected: OptimizationProposal | None = None
    applied: AppliedOptimization | None = None
    # half_auto 模式下呈现待批准的干预请求（需求 7.3）
    approval_request: InterventionRequest | None = None


# ---- 源码改动接口（副作用经注入，测试可替换 fake）----
@runtime_checkable
class SourceEditor(Protocol):
    """把一项优化提议落到 C 源码的抽象接口。

    真实实现改动磁盘上的源码文件；测试可注入 fake 以断言「提议 → 改动」映射而不
    触碰真实源码树。返回一段人类可读的改动摘要（写入 ``AppliedOptimization``）。
    """

    def apply(self, proposal: OptimizationProposal) -> str: ...


class FileSourceEditor:
    """基于文件系统的 :class:`SourceEditor` 默认实现：文本片段替换。

    在 ``root`` 下定位 ``proposal.file_path``，将 ``original_snippet`` 的首次出现
    替换为 ``optimized_snippet`` 并写回。找不到文件或片段时抛出清晰错误，避免静默
    产生错误改动。
    """

    def __init__(self, root: str | Path = ".") -> None:
        self._root = Path(root)

    def apply(self, proposal: OptimizationProposal) -> str:
        target = self._root / proposal.file_path
        if not target.is_file():
            raise FileNotFoundError(f"待优化的源码文件不存在：{target}")
        text = target.read_text(encoding="utf-8")
        if proposal.original_snippet and proposal.original_snippet not in text:
            raise ValueError(
                f"在 {proposal.file_path} 中未找到待替换的原始代码片段，无法安全应用优化。"
            )
        updated = text.replace(
            proposal.original_snippet, proposal.optimized_snippet, 1
        )
        target.write_text(updated, encoding="utf-8")
        return (
            f"已将 {proposal.file_path} 中函数 {proposal.target_func} 的实现替换为"
            f"优化版本：{proposal.title}"
        )


_DEFAULT_SYSTEM_PROMPT = (
    "你是嵌入式 LVGL 性能调优的优化器（Optimizer）子智能体。给定一个已隔离的叶子级"
    "热点函数及其证据，请提出一项或多项源码级优化。每一项都必须：\n"
    "1) 明确引用目标函数（target_func）；\n"
    "2) 说明预期的性能效果（expected_effect，尽量量化，如减少多少微秒/降低调用次数）；\n"
    "3) 给出可直接应用的最小代码改动（file_path、original_snippet、optimized_snippet）。\n"
    "只针对给定热点提出优化，不要泛泛而谈。"
)


class Optimizer:
    """Optimizer 子智能体：提出优化，并按自动化模式应用或呈现待批准。

    副作用全部经注入接口完成（设计原则「一切副作用皆经接口注入」）：

    * ``llm``：:class:`BaseLLMProvider`，驱动优化提议这一推理步骤（需求 7.1）；
    * ``device``：:class:`DeviceIO`，``full_auto`` 下重编译 + 烧录（需求 7.2）；
    * ``source_editor``：:class:`SourceEditor`，把提议落到 C 源码；缺省用
      :class:`FileSourceEditor`。
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        device: DeviceIO,
        *,
        source_editor: SourceEditor | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self._device = device
        self._source_editor: SourceEditor = source_editor or FileSourceEditor()
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    # -- 需求 7.1：提出优化（LLM 推理步骤）--
    def propose(
        self,
        hotspot: HotspotEntry,
        *,
        report: FrameReport | None = None,
        context: str = "",
    ) -> list[OptimizationProposal]:
        """针对叶子热点提出一项或多项优化，每项引用目标函数与预期效果（需求 7.1）。"""
        model = self._llm.get_chat_model().with_structured_output(
            OptimizationProposalList
        )
        human = self._build_prompt(hotspot, report=report, context=context)
        result = model.invoke(
            [SystemMessage(content=self._system_prompt), HumanMessage(content=human)]
        )
        proposals = list(getattr(result, "proposals", []) or [])
        # 兜底：确保每项都引用到目标函数（需求 7.1）。
        for p in proposals:
            if not p.target_func:
                p.target_func = hotspot.func
        return proposals

    def _build_prompt(
        self,
        hotspot: HotspotEntry,
        *,
        report: FrameReport | None,
        context: str,
    ) -> str:
        """把热点证据与上下文拼为提议请求（原始日志绝不进入，仅小体量摘要）。"""
        lines = [
            "已隔离的叶子级热点：",
            f"- 函数：{hotspot.func}",
            f"- 慢帧内聚合自耗时：{hotspot.total_us} us",
            f"- 调用次数：{hotspot.call_count}",
            f"- 热点排名：{hotspot.rank}",
        ]
        if report is not None:
            lines += [
                "",
                "所属场景分析摘要：",
                f"- 场景：{report.scenario}",
                f"- 帧预算：{report.frame_budget_us} us（目标 {report.target_fps} FPS）",
                f"- 慢帧数：{len(report.slow_frames)} / 总帧数 {report.total_frames}",
            ]
            if report.summary:
                lines.append(f"- 瓶颈概括：{report.summary}")
        if context:
            lines += ["", "补充上下文：", context]
        return "\n".join(lines)

    # -- 需求 7.2 / 7.3：按模式应用或呈现待批准 --
    def run(
        self,
        hotspot: HotspotEntry | None,
        mode: Literal["full_auto", "half_auto"],
        *,
        report: FrameReport | None = None,
        context: str = "",
    ) -> OptimizerResult:
        """提出优化，并依据自动化模式应用（full_auto）或呈现待批准（half_auto）。

        * ``hotspot is None``：无叶子热点，返回 ``no_hotspot``。
        * ``full_auto``：选取首项提议直接应用 + 重编译 + 烧录（需求 7.2）。
        * ``half_auto``：仅呈现提议、返回 ``pending_approval`` 与干预请求，
          在开发者批准前不应用任何改动（需求 7.3）。
        """
        if hotspot is None:
            return OptimizerResult(status="no_hotspot")

        proposals = self.propose(hotspot, report=report, context=context)

        if not proposals:
            # 无可行提议：等同于无可应用对象，交回 Orchestrator 决策。
            return OptimizerResult(status="no_hotspot", proposals=[])

        if mode == "half_auto":
            # 需求 7.3：呈现待批准，不应用。
            request = InterventionRequest(
                action="approve_optimization",
                instruction=(
                    "请审阅以下针对叶子热点的优化提议，批准后系统方可应用并重编译烧录。"
                ),
                context={
                    "target_func": hotspot.func,
                    "proposals": [p.model_dump() for p in proposals],
                },
            )
            return OptimizerResult(
                status="pending_approval",
                proposals=proposals,
                approval_request=request,
            )

        # 需求 7.2：full_auto 直接应用首项提议 + 重编译 + 烧录。
        selected = proposals[0]
        applied = self.apply(selected)
        return OptimizerResult(
            status="applied",
            proposals=proposals,
            selected=selected,
            applied=applied,
        )

    def apply(self, proposal: OptimizationProposal) -> AppliedOptimization:
        """把一项优化落到 C 源码，随后重编译并烧录（需求 7.2 / 7.4）。

        在 ``half_auto`` 后端下，``device.build()`` / ``flash()`` 可能返回
        :class:`InterventionRequest`（不可脚本化时降级为人在环，需求 14.2）；
        此处分别归入结果对象，供 Orchestrator 在干预点暂停等待人工完成。
        """
        summary = self._source_editor.apply(proposal)

        applied = AppliedOptimization(
            target_func=proposal.target_func,
            file_path=proposal.file_path,
            summary=summary,
        )

        build = self._device.build()
        if isinstance(build, InterventionRequest):
            applied.build_intervention = build
        else:
            applied.build = build

        flash = self._device.flash()
        if isinstance(flash, InterventionRequest):
            applied.flash_intervention = flash
        else:
            applied.flash = flash

        return applied
