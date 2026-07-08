"""DeviceIO 相关的辅助 Pydantic v2 数据契约。

design.md 的 "Data Models" 章节聚焦跨层通用契约（``core/models.py``），并未展开
设备控制面的构建/烧录/输入/人在环等结果类型。这些类型是 ``DeviceIO`` 接口
（design.md "Components and Interfaces → 基础设施层 → DeviceIO"）的直接组成部分，
故就近定义于 device 模块内，保持与设计一致：

* ``BuildResult`` / ``FlashResult``：``build()`` / ``flash()`` 的结构化结果，
  以结果对象（而非裸异常）建模，便于确定性断言与 API 序列化
  （见 design.md "Error Handling" 收束原则）。
* ``InputEvent``：``inject_input(event)`` 注入的一次输入事件，用以触发待测场景。
* ``InterventionRequest``：当 build/flash/inject 无法脚本化时，``HumanInLoopMixin``
  返回它以请求人工介入，而非抛错——供 Half_Auto_Mode 使用（需求 14.2）。

全部可独立构造、可无损 JSON 序列化（支撑测试与语言无关 API 边界，Property 9）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "InputEvent",
    "BuildResult",
    "FlashResult",
    "InterventionRequest",
]


class InputEvent(BaseModel):
    """一次待注入设备的输入事件，用以触发被测场景（需求 2.6 / 19.2）。

    ``kind`` 描述事件类别（如 ``"touch"`` / ``"key"`` / ``"gesture"``），
    ``params`` 承载与该类别相关的参数（坐标、键值、时长等），保持后端无关。
    """

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class BuildResult(BaseModel):
    """``DeviceIO.build()`` 的结构化结果。

    ``success`` 为编译是否成功；``output`` 保留构建日志摘要；成功时
    ``artifact_path`` 可指向产物（固件/镜像）路径；失败时 ``error`` 给出原因。
    """

    success: bool
    output: str = ""
    artifact_path: str | None = None
    error: str | None = None


class FlashResult(BaseModel):
    """``DeviceIO.flash()`` 的结构化结果。

    ``success`` 为烧录是否成功；``output`` 保留烧录日志摘要；失败时
    ``error`` 给出原因。
    """

    success: bool
    output: str = ""
    error: str | None = None


class InterventionRequest(BaseModel):
    """人在环干预请求：某个操作无法脚本化，需人工完成后再继续。

    ``HumanInLoopMixin`` 在 build/flash/inject 不可自动执行时返回该对象而非
    抛错（需求 14.2）：``action`` 标记被降级的操作名，``instruction`` 是给
    操作者的可读指引，``context`` 携带辅助信息（如待注入事件、构建命令等）。
    """

    action: str
    instruction: str
    context: dict[str, Any] = Field(default_factory=dict)
