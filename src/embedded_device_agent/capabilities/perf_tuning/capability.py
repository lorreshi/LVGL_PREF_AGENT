"""``PerfTuningCapability``：把 Capability A 调优闭环封装为可插拔能力（任务 14.2）。

严格对照 design.md「平台层」与需求 16.4/16.5：实现 :class:`BaseCapability` 契约并经
``@CapabilityFactory.register("perf_tuning")`` 注册，使平台无需改动即可发现、构造并
路由到本能力（Property 10）。

* :meth:`build_graph` 复用任务 12.3/12.4 的 :func:`build_tuning_graph`，用注入的共享
  底座 :class:`CoreServices` 构建已编译的调优闭环 subgraph（需求 16.4）。
* :meth:`as_tool` 把该 subgraph 暴露为一个 LangChain 工具（subgraph-as-tool，
  需求 16.4/16.5）：Router 依 ``description`` 分发到本工具后，工具以给定场景启动一次
  调优运行并回传小体量的结果摘要（不外泄原始日志，Property 12）。
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from embedded_device_agent.capabilities.base import BaseCapability
from embedded_device_agent.capabilities.factory import CapabilityFactory
from embedded_device_agent.capabilities.perf_tuning.graph import build_tuning_graph
from embedded_device_agent.core.services import CoreServices

__all__ = ["PerfTuningCapability"]


@CapabilityFactory.register("perf_tuning")
class PerfTuningCapability(BaseCapability):
    """LVGL 嵌入式性能调优能力（Capability A）。"""

    name = "perf_tuning"
    description = (
        "LVGL 嵌入式设备性能调优：采集串口 profiler trace，检测超预算慢帧与热点函数，"
        "下钻埋点并提出/应用源码级优化，闭环复测验证并沉淀调优知识。"
        "适用于卡顿、掉帧、帧率不达标、渲染耗时过高等性能问题。"
    )

    def build_graph(self, core: CoreServices) -> CompiledStateGraph:
        """用共享底座构建调优闭环 subgraph（需求 16.4）。"""
        return build_tuning_graph(core)

    def as_tool(self, core: CoreServices) -> BaseTool:
        """把调优闭环暴露为 subgraph-as-tool（需求 16.4/16.5）。

        工具入参为待调优场景描述（及可选自动化模式）；内部以独立 ``thread_id`` 启动
        一次调优运行，回传结果摘要（瓶颈概括 / 是否达标 / 已应用优化），原始日志不外泄。
        """
        graph = self.build_graph(core)
        default_mode = core.config.mode

        @tool("perf_tuning")
        def perf_tuning_tool(scenario: str, mode: str | None = None) -> dict[str, Any]:
            """对给定场景运行一次 LVGL 性能调优闭环并返回结果摘要。

            Args:
                scenario: 待复现/调优的场景描述（如"滚动列表卡顿"）。
                mode: 自动化模式 ``full_auto`` / ``half_auto``；缺省取配置值。
            """
            thread_id = f"perf_tuning-{uuid.uuid4().hex}"
            config = {
                "recursion_limit": 500,
                "configurable": {"thread_id": thread_id},
            }
            final = graph.invoke(
                {"scenario": scenario, "mode": mode or default_mode},
                config=config,
            )
            report = final.get("latest")
            return {
                "thread_id": thread_id,
                "done": bool(final.get("done")),
                "iterations": final.get("iteration", 0),
                "summary": getattr(report, "summary", None),
                "no_slow_frames": getattr(report, "no_slow_frames", None),
                "optimizations_applied": final.get("optimizations_applied", []),
            }

        return perf_tuning_tool
