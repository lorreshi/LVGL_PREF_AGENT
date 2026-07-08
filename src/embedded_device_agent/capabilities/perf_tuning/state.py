"""调优闭环的 LangGraph 运行状态（``TuningRunState``）。

该 ``TypedDict`` 是 LangGraph subgraph 在各节点间传递、并由 Checkpointer
持久化的运行状态。字段严格对照 design.md "Data Models" 章节。

关键约束（Property 12：原始日志不进上下文）：

* ``latest_artifact`` 只保存 ``ArtifactRef`` 引用（run_id + kind + path），
  绝不内联原始日志 / systrace / 调用树内容；大工件落盘，state 保持小体量，
  从而 Checkpointer 持久化的状态也保持小体量。
* ``baseline`` / ``latest`` 只保存 ``FrameReport`` 小体量摘要。

自动化模式与迭代边界（需求 11.1）：

* ``mode`` 为 ``"full_auto"`` 或 ``"half_auto"``；
* ``iteration`` 为当前轮次，``decide`` 每轮检查 ``iteration < max_iterations``，
  超限走 ``halt`` 收尾（需求 6.5 / 11.5）。
"""

from __future__ import annotations

from typing import Literal, TypedDict

from embedded_device_agent.core.models import (
    ArtifactRef,
    FrameReport,
    KnowledgeRecord,
)

__all__ = ["TuningRunState"]


class TuningRunState(TypedDict):
    """调优闭环运行状态（LangGraph State）。"""

    thread_id: str
    scenario: str
    mode: Literal["full_auto", "half_auto"]
    iteration: int
    max_iterations: int
    latest_artifact: ArtifactRef | None  # 仅存引用，日志内容不进 state/context
    baseline: FrameReport | None  # 小体量摘要
    latest: FrameReport | None
    instrumentation_history: list[str]
    optimizations_applied: list[str]
    recalled_knowledge: list[KnowledgeRecord]
    done: bool
