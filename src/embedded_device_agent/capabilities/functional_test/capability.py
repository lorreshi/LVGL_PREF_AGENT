"""``FunctionalTestCapability``：把功能测试骨架封装为可插拔能力（任务 15.2）。

对照 design.md「平台层」与需求 19.1/19.4/18.2：实现 :class:`BaseCapability` 契约并经
``@CapabilityFactory.register("functional_test")`` 注册，证明平台的可插拔契约对**第二个
能力**同样成立（Property 10），且共享 DeviceIO 被本能力复用（需求 18.2）。

* :meth:`build_graph` 复用任务 15.2 的 :func:`build_functional_test_graph` 构建最小可跑通
  的功能测试 subgraph。
* :meth:`as_tool` 把该 subgraph 暴露为 subgraph-as-tool：入参为一个用例（步骤列表），
  运行后回传语言中立的测试报告摘要。
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from embedded_device_agent.capabilities.base import BaseCapability
from embedded_device_agent.capabilities.factory import CapabilityFactory
from embedded_device_agent.capabilities.functional_test.graph import (
    build_functional_test_graph,
)
from embedded_device_agent.capabilities.functional_test.models import TestCase
from embedded_device_agent.core.services import CoreServices

__all__ = ["FunctionalTestCapability"]


@CapabilityFactory.register("functional_test")
class FunctionalTestCapability(BaseCapability):
    """嵌入式设备自动化功能测试能力（Capability B，最小骨架）。"""

    name = "functional_test"
    description = (
        "嵌入式设备自动化功能测试：按测试用例经串口向设备注入输入/发送命令并校验响应，"
        "产出通过/失败的功能测试报告。适用于验证设备功能行为、命令响应、回归测试等。"
    )

    def build_graph(self, core: CoreServices) -> CompiledStateGraph:
        """用共享底座构建功能测试最小骨架 subgraph（需求 19.1/18.2）。"""
        return build_functional_test_graph(core)

    def as_tool(self, core: CoreServices) -> BaseTool:
        """把功能测试骨架暴露为 subgraph-as-tool（需求 16.4）。"""
        graph = self.build_graph(core)

        @tool("functional_test")
        def functional_test_tool(case: dict[str, Any]) -> dict[str, Any]:
            """按给定测试用例运行一次功能测试并返回报告摘要。

            Args:
                case: 测试用例（``TestCase`` 的 JSON 形态，含 ``name`` 与 ``steps``）。
            """
            test_case = TestCase.model_validate(case)
            thread_id = f"functional_test-{uuid.uuid4().hex}"
            config = {"configurable": {"thread_id": thread_id}}
            final = graph.invoke({"case": test_case}, config=config)
            report = final.get("report")
            return {
                "thread_id": thread_id,
                "done": bool(final.get("done")),
                "report": report.model_dump(mode="json") if report is not None else None,
            }

        return functional_test_tool
