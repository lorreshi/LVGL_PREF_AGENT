"""Capability B（FunctionalTest）数据契约（任务 15.1）。

对照 design.md「Capability B 契约/骨架」与需求 19.3：定义测试用例、断言、测试报告的
Pydantic v2 契约。所有模型可独立构造、可无损 JSON 序列化，从而经语言无关的 API 边界
中立暴露（与 Capability A 的工件契约一致，Property 9 的精神）。

一次功能测试的基本形态：

* :class:`TestStep`——一个可脚本化的设备动作（经共享 DeviceIO：``inject_input`` 注入
  输入、``send_cmd`` 发命令），可携带一条对设备响应的 :class:`Assertion`。
* :class:`TestCase`——若干有序步骤构成的一个用例。
* :class:`Assertion` / :class:`AssertionResult`——期望与实判结果。
* :class:`StepResult` / :class:`TestReport`——单步与整例的执行结论。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "Assertion",
    "TestStep",
    "TestCase",
    "AssertionResult",
    "StepResult",
    "TestReport",
]

# 支持的动作类型：注入输入事件 / 发送命令 / 采集一段输出。
ActionKind = Literal["inject_input", "send_cmd", "capture"]
# 支持的断言算子。
AssertionOp = Literal["equals", "contains", "not_contains", "regex"]


class Assertion(BaseModel):
    """对某一步骤设备响应的断言（期望）。"""

    op: AssertionOp = "contains"
    expected: str = ""
    # 断言描述，便于报告可读。
    description: str = ""


class TestStep(BaseModel):
    """一个可脚本化的测试步骤：经共享 DeviceIO 驱动设备执行一个动作。"""

    __test__ = False  # 避免 pytest 误采集为测试类

    name: str
    action: ActionKind
    # 动作载荷：send_cmd 用 ``command``；inject_input 用 ``input_kind`` / ``input_params``；
    # capture 用 ``duration_s``。未用到的字段留空即可。
    command: str = ""
    input_kind: str = ""
    input_params: dict[str, Any] = Field(default_factory=dict)
    duration_s: float = 1.0
    # 可选断言：对本步骤设备响应做校验。
    assertion: Assertion | None = None


class TestCase(BaseModel):
    """一个功能测试用例：一组有序步骤。"""

    __test__ = False  # 避免 pytest 误采集为测试类

    name: str
    description: str = ""
    steps: list[TestStep] = Field(default_factory=list)


class AssertionResult(BaseModel):
    """一条断言的实判结果。"""

    op: AssertionOp
    expected: str
    actual: str
    passed: bool
    description: str = ""


class StepResult(BaseModel):
    """单个步骤的执行结论。"""

    name: str
    action: ActionKind
    response: str = ""
    assertion: AssertionResult | None = None
    # 步骤是否通过：无断言的步骤只要执行未报错即视为通过。
    passed: bool = True
    error: str | None = None


class TestReport(BaseModel):
    """一次功能测试运行的整例报告（语言中立、可 JSON 序列化，需求 19.3）。"""

    __test__ = False  # 避免 pytest 误采集为测试类

    case_name: str
    passed: bool
    total_steps: int
    passed_steps: int
    failed_steps: int
    steps: list[StepResult] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    summary: str = ""
