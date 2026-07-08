"""Capability B（FunctionalTest）最小可跑通骨架图（任务 15.2）。

对照 design.md「Capability B 契约/骨架」与需求 19.1/19.2/19.4/18.2：用 LangGraph 构建
一个**最小可端到端跑通**的功能测试流程，全程经**共享 DeviceIO**（``inject_input`` /
``send_cmd`` / ``capture``）驱动设备——从而证明底座（尤其 DeviceIO）被第二个能力复用
（需求 18.2），且平台契约对新能力成立（Property 10）。

流程：``[*] → setup → run_steps → report → [*]``。``run_steps`` 顺序执行用例的每个步骤，
对带断言的步骤校验设备响应，累计 :class:`StepResult`；``report`` 汇总为
:class:`TestReport`。副作用经注入的 DeviceIO 完成，测试时以 ``FakeDeviceIO`` 替换即可
离线跑通（需求 20）。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from embedded_device_agent.capabilities.functional_test.models import (
    Assertion,
    AssertionResult,
    StepResult,
    TestCase,
    TestReport,
    TestStep,
)
from embedded_device_agent.core.device.base import DeviceIO
from embedded_device_agent.core.device.models import InputEvent
from embedded_device_agent.core.services import CoreServices

__all__ = ["FunctionalTestState", "build_functional_test_graph"]


class FunctionalTestState(TypedDict, total=False):
    """功能测试运行状态。"""

    case: TestCase
    step_results: list[StepResult]
    report: TestReport | None
    done: bool


def _eval_assertion(assertion: Assertion, actual: str) -> AssertionResult:
    """按算子校验设备响应，返回断言结果。"""
    exp = assertion.expected
    if assertion.op == "equals":
        passed = actual == exp
    elif assertion.op == "contains":
        passed = exp in actual
    elif assertion.op == "not_contains":
        passed = exp not in actual
    elif assertion.op == "regex":
        passed = re.search(exp, actual) is not None
    else:  # pragma: no cover - Literal 已约束取值
        passed = False
    return AssertionResult(
        op=assertion.op,
        expected=exp,
        actual=actual,
        passed=passed,
        description=assertion.description,
    )


def build_functional_test_graph(
    services: CoreServices,
    *,
    device: DeviceIO | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """构建并编译功能测试最小骨架图（需求 19）。

    Parameters
    ----------
    services:
        共享底座；设备经 ``services.device`` 复用（需求 18.2），亦可用 ``device`` 覆盖注入。
    device:
        可选覆盖的 DeviceIO（测试可注入 FakeDeviceIO）。
    checkpointer:
        可选 Checkpointer；缺省用 ``services.checkpointer``。
    """
    dev = device or services.device
    cfg = services.config

    def setup(state: FunctionalTestState) -> dict[str, Any]:
        """打开设备连接（若配置了端口），准备执行用例。"""
        if cfg.device.port:
            dev.open_serial(cfg.device.port, cfg.device.baud)
        return {"step_results": []}

    def run_steps(state: FunctionalTestState) -> dict[str, Any]:
        """顺序执行用例步骤，经共享 DeviceIO 驱动设备并校验断言（需求 19.2）。"""
        case = state["case"]
        results: list[StepResult] = []
        for step in case.steps:
            results.append(_run_step(dev, step))
        return {"step_results": results}

    def report(state: FunctionalTestState) -> dict[str, Any]:
        """汇总步骤结果为整例 TestReport（需求 19.4）。"""
        case = state["case"]
        results = state.get("step_results", [])
        passed_steps = sum(1 for r in results if r.passed)
        failed_steps = len(results) - passed_steps
        now = datetime.now(timezone.utc)
        test_report = TestReport(
            case_name=case.name,
            passed=failed_steps == 0,
            total_steps=len(results),
            passed_steps=passed_steps,
            failed_steps=failed_steps,
            steps=results,
            started_at=now,
            finished_at=now,
            summary=(
                f"用例「{case.name}」{'通过' if failed_steps == 0 else '失败'}："
                f"{passed_steps}/{len(results)} 步通过。"
            ),
        )
        return {"report": test_report, "done": True}

    graph: StateGraph = StateGraph(FunctionalTestState)
    graph.add_node("setup", setup)
    graph.add_node("run_steps", run_steps)
    graph.add_node("report", report)
    graph.add_edge(START, "setup")
    graph.add_edge("setup", "run_steps")
    graph.add_edge("run_steps", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer or services.checkpointer)


def _run_step(dev: DeviceIO, step: TestStep) -> StepResult:
    """执行单个步骤：经共享 DeviceIO 施加动作并（若有）校验断言。"""
    response = ""
    try:
        if step.action == "send_cmd":
            response = dev.send_cmd(step.command)
        elif step.action == "inject_input":
            dev.inject_input(InputEvent(kind=step.input_kind, params=step.input_params))
        elif step.action == "capture":
            artifact = dev.capture(step.duration_s)
            response = str(artifact.path)
    except Exception as exc:  # noqa: BLE001 - 步骤级错误如实记录，不中断整例
        return StepResult(
            name=step.name,
            action=step.action,
            response=response,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    assertion_result: AssertionResult | None = None
    passed = True
    if step.assertion is not None:
        assertion_result = _eval_assertion(step.assertion, response)
        passed = assertion_result.passed

    return StepResult(
        name=step.name,
        action=step.action,
        response=response,
        assertion=assertion_result,
        passed=passed,
    )
