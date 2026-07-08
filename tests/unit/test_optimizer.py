"""任务 11.9：Optimizer 子智能体单元测试（需求 7.1 / 7.2 / 7.3 / 7.4）。

对照 optimizer.py 的真实 API（``propose`` / ``run`` / ``apply``）覆盖：

* **需求 7.1**：优化提议结构——每项引用目标函数（``target_func``）与预期效果
  （``expected_effect``）；``propose`` 对缺失 ``target_func`` 的提议以热点函数兜底。
* **需求 7.2 / 7.4**：``full_auto`` 分支应用首项提议（经注入的 fake ``SourceEditor``）、
  经 ``FakeDeviceIO`` 重编译 + 烧录，返回 ``status="applied"`` 且 ``AppliedOptimization``
  记录目标函数与改动摘要。
* **需求 7.3**：``half_auto`` 分支返回 ``status="pending_approval"`` 与审批干预请求，
  且不应用任何源码改动。
* ``hotspot is None`` 返回 ``no_hotspot``。

副作用全部经注入接口替换为 fake：LLM 结构化输出经桩 provider、源码改动经 fake
``SourceEditor``、编译烧录经 ``FakeDeviceIO``——不触达真实源码树 / 硬件 / LLM。

``optimizer.propose()`` 依赖 ``llm.get_chat_model().with_structured_output(...)``，而
harness 的 ``FakeLLMProvider`` 底层 ``FakeListChatModel`` **不支持**
``with_structured_output``（会抛 ``NotImplementedError``）。故此处注入一个最小桩
provider，其 chat model 的 ``with_structured_output`` 返回一个直接回放预置
``OptimizationProposalList`` 的可调用对象。
"""

from __future__ import annotations

from embedded_device_agent.capabilities.perf_tuning.agents.optimizer import (
    AppliedOptimization,
    OptimizationProposal,
    OptimizationProposalList,
    Optimizer,
    OptimizerResult,
    SourceEditor,
)
from embedded_device_agent.core.device.models import InterventionRequest
from embedded_device_agent.core.llm.base import BaseLLMProvider
from embedded_device_agent.core.models import HotspotEntry
from tests.harness import FakeDeviceIO


# --------------------------------------------------------------------------- #
# 测试替身（stub / fake）
# --------------------------------------------------------------------------- #
class _StructuredOutputModel:
    """桩：回放预置 ``OptimizationProposalList``，模拟结构化输出可运行体。"""

    def __init__(self, proposals: OptimizationProposalList) -> None:
        self._proposals = proposals

    def invoke(self, _messages: object) -> OptimizationProposalList:
        return self._proposals


class _StubChatModel:
    """桩 chat model：``with_structured_output`` 返回回放结构化提议的可运行体。

    ``FakeListChatModel`` 不支持 ``with_structured_output``，故用本桩替代，聚焦
    Optimizer 自身逻辑而非 LangChain 的结构化输出实现。
    """

    def __init__(self, proposals: OptimizationProposalList) -> None:
        self._proposals = proposals

    def with_structured_output(self, _schema: object) -> _StructuredOutputModel:
        return _StructuredOutputModel(self._proposals)


class _StubLLMProvider(BaseLLMProvider):
    """最小 ``BaseLLMProvider``：其 chat model 回放预置结构化提议。"""

    def __init__(self, proposals: list[OptimizationProposal]) -> None:
        self._proposals = OptimizationProposalList(proposals=proposals)

    @property
    def name(self) -> str:
        return "stub-structured"

    def get_chat_model(self) -> _StubChatModel:  # type: ignore[override]
        return _StubChatModel(self._proposals)


class _RecordingSourceEditor:
    """记录被应用提议的 fake ``SourceEditor``，不触碰任何真实源码文件。"""

    def __init__(self) -> None:
        self.applied: list[OptimizationProposal] = []

    def apply(self, proposal: OptimizationProposal) -> str:
        self.applied.append(proposal)
        return f"fake-edit: {proposal.target_func} -> {proposal.title}"


# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #
def _hotspot(func: str = "lv_draw_rect") -> HotspotEntry:
    return HotspotEntry(func=func, total_us=12_000, call_count=48, rank=1)


def _proposal(
    *,
    target_func: str = "lv_draw_rect",
    title: str = "缓存圆角遮罩",
    expected_effect: str = "每帧减少约 8000us 的重复遮罩计算",
) -> OptimizationProposal:
    return OptimizationProposal(
        target_func=target_func,
        title=title,
        description="将逐帧重算的圆角遮罩改为一次性缓存复用。",
        expected_effect=expected_effect,
        file_path="src/ui/draw_rect.c",
        original_snippet="compute_mask();",
        optimized_snippet="use_cached_mask();",
        rationale="热点证据显示遮罩计算占据主导自耗时。",
    )


def _optimizer(
    proposals: list[OptimizationProposal],
    *,
    editor: SourceEditor | None = None,
    device: FakeDeviceIO | None = None,
) -> tuple[Optimizer, _RecordingSourceEditor, FakeDeviceIO]:
    ed = editor or _RecordingSourceEditor()
    dev = device or FakeDeviceIO()
    opt = Optimizer(_StubLLMProvider(proposals), dev, source_editor=ed)
    return opt, ed, dev


# --------------------------------------------------------------------------- #
# 需求 7.1：优化提议结构（引用目标函数 + 预期效果）
# --------------------------------------------------------------------------- #
def test_propose_returns_proposals_with_target_func_and_expected_effect() -> None:
    """每项提议都引用目标函数并给出预期性能效果（需求 7.1）。"""
    opt, _, _ = _optimizer([_proposal()])

    proposals = opt.propose(_hotspot())

    assert len(proposals) == 1
    p = proposals[0]
    assert isinstance(p, OptimizationProposal)
    assert p.target_func == "lv_draw_rect"
    assert p.expected_effect  # 预期效果非空
    assert p.expected_effect == "每帧减少约 8000us 的重复遮罩计算"


def test_propose_supports_multiple_proposals() -> None:
    """需求 7.1「一项或多项」：可一次返回多项提议。"""
    opt, _, _ = _optimizer(
        [
            _proposal(title="缓存圆角遮罩"),
            _proposal(title="减少无效重绘", expected_effect="降低 30% 调用次数"),
        ]
    )

    proposals = opt.propose(_hotspot())

    assert [p.title for p in proposals] == ["缓存圆角遮罩", "减少无效重绘"]
    assert all(p.target_func == "lv_draw_rect" for p in proposals)
    assert all(p.expected_effect for p in proposals)


def test_propose_backfills_missing_target_func_from_hotspot() -> None:
    """提议若缺 target_func，则以热点函数兜底，确保始终引用目标函数（需求 7.1）。"""
    opt, _, _ = _optimizer([_proposal(target_func="")])

    proposals = opt.propose(_hotspot(func="lv_refr_area"))

    assert proposals[0].target_func == "lv_refr_area"


# --------------------------------------------------------------------------- #
# 需求 7.2 / 7.4：full_auto 分支应用提议 + 重编译烧录 + 状态记录
# --------------------------------------------------------------------------- #
def test_full_auto_applies_proposal_and_rebuilds_and_flashes() -> None:
    """full_auto 应用首项提议、经 DeviceIO 重编译并烧录，返回 applied（需求 7.2）。"""
    proposal = _proposal()
    opt, editor, device = _optimizer([proposal])

    result = opt.run(_hotspot(), mode="full_auto")

    assert isinstance(result, OptimizerResult)
    assert result.status == "applied"
    assert result.selected == proposal

    # 源码改动经注入的 fake editor 落地（未触碰真实源码）
    assert editor.applied == [proposal]

    # 经 FakeDeviceIO 重编译 + 烧录（需求 7.2）
    assert ("build", None) in device.calls
    assert ("flash", None) in device.calls


def test_full_auto_records_applied_optimization_with_target_and_summary() -> None:
    """已应用优化记录目标函数与改动摘要，供 Orchestrator 写入运行状态（需求 7.4）。"""
    proposal = _proposal(target_func="lv_draw_rect")
    opt, _, _ = _optimizer([proposal])

    result = opt.run(_hotspot(), mode="full_auto")

    applied = result.applied
    assert isinstance(applied, AppliedOptimization)
    assert applied.target_func == "lv_draw_rect"  # 需求 7.4：记录目标函数
    assert applied.summary  # 需求 7.4：记录改动摘要
    assert "lv_draw_rect" in applied.summary
    # 构建 / 烧录结果被记录且成功
    assert applied.build is not None and applied.build.success is True
    assert applied.flash is not None and applied.flash.success is True
    # 未降级为人在环干预
    assert applied.build_intervention is None
    assert applied.flash_intervention is None


def test_apply_maps_proposal_to_applied_optimization() -> None:
    """apply() 直接把提议落地并返回记录，涵盖目标函数/文件/摘要（需求 7.4）。"""
    proposal = _proposal()
    opt, editor, device = _optimizer([proposal])

    applied = opt.apply(proposal)

    assert applied.target_func == proposal.target_func
    assert applied.file_path == proposal.file_path
    assert editor.applied == [proposal]
    assert ("build", None) in device.calls and ("flash", None) in device.calls


# --------------------------------------------------------------------------- #
# 需求 7.3：half_auto 分支呈现待批准且不应用
# --------------------------------------------------------------------------- #
def test_half_auto_returns_pending_approval_and_applies_nothing() -> None:
    """half_auto 呈现审批干预请求、不应用任何改动、不重编译烧录（需求 7.3）。"""
    proposal = _proposal()
    opt, editor, device = _optimizer([proposal])

    result = opt.run(_hotspot(), mode="half_auto")

    assert result.status == "pending_approval"
    assert result.proposals == [proposal]
    assert result.applied is None
    assert result.selected is None

    # 审批干预请求呈现待批准的提议（需求 7.3）
    req = result.approval_request
    assert isinstance(req, InterventionRequest)
    assert req.action == "approve_optimization"
    assert req.context["target_func"] == "lv_draw_rect"
    assert req.context["proposals"] == [proposal.model_dump()]

    # 关键：批准前不应用任何源码改动、不重编译/烧录
    assert editor.applied == []
    assert ("build", None) not in device.calls
    assert ("flash", None) not in device.calls


# --------------------------------------------------------------------------- #
# 边界：无叶子热点 / 无可行提议
# --------------------------------------------------------------------------- #
def test_run_with_no_hotspot_returns_no_hotspot() -> None:
    """未提供叶子热点时返回 no_hotspot，不产生任何副作用。"""
    opt, editor, device = _optimizer([_proposal()])

    result = opt.run(None, mode="full_auto")

    assert result.status == "no_hotspot"
    assert result.proposals == []
    assert editor.applied == []
    assert device.calls == []


def test_run_with_empty_proposals_returns_no_hotspot() -> None:
    """LLM 无可行提议时等同于无可应用对象，返回 no_hotspot 且不应用。"""
    opt, editor, device = _optimizer([])

    result = opt.run(_hotspot(), mode="full_auto")

    assert result.status == "no_hotspot"
    assert editor.applied == []
    assert ("build", None) not in device.calls
