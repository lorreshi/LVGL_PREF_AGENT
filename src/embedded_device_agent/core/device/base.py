"""DeviceIO 基类与 HumanInLoopMixin。

严格对照 design.md "Components and Interfaces → 基础设施层 → DeviceIO"：

``DeviceIO`` 是对整个**设备控制面**的抽象——串口开合、trace 采集、编译、烧录、
输入注入、命令往返，是所有能力复用的核心副作用边界（需求 14.1 / 18.2）。所有
设备副作用都藏在该基类 + 工厂之后，由 YAML 的 ``type`` 字段选择具体实现
（需求 12.4）；测试时以 ``FakeDeviceIO`` 替换即可脱离真实硬件驱动完整 subgraph。

``HumanInLoopMixin`` 供半自动后端复用：当 build/flash/inject 无法脚本化时，转为
返回一个 ``InterventionRequest`` 而非抛错，供 Half_Auto_Mode 在干预点暂停等待人工
完成（需求 14.2）。

具体后端实现（SerialDeviceIO / HalfAutoDeviceIO / FakeDeviceIO）在任务 4.2 / 4.3
及测试 harness 中提供，并经 ``@DeviceIOFactory.register(...)`` 注册。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from embedded_device_agent.core.device.models import (
    BuildResult,
    FlashResult,
    InputEvent,
    InterventionRequest,
)
from embedded_device_agent.core.models import RawTraceArtifact

__all__ = ["DeviceIO", "HumanInLoopMixin"]


class DeviceIO(ABC):
    """设备控制面的统一抽象契约（需求 14.1）。

    子类通常在 ``__init__`` 中接收一个 ``DeviceConfig``，据此惰性持有底层资源
    （串口句柄、构建/烧录命令等）。真实副作用发生在具体子类内部，基类只声明契约。
    """

    @abstractmethod
    def open_serial(self, port: str, baud: int) -> None:
        """打开串口连接。

        打开前应校验 ``baud <= max_safe_baud``（需求 2.3）；无法打开时返回带端口
        与原因的描述性错误（需求 2.4）。
        """
        raise NotImplementedError

    @abstractmethod
    def capture(self, duration_s: float) -> RawTraceArtifact:
        """采集 ``duration_s`` 秒的原始 trace 日志，落盘并返回其工件引用。"""
        raise NotImplementedError

    @abstractmethod
    def build(self) -> BuildResult:
        """触发固件编译，返回结构化构建结果。"""
        raise NotImplementedError

    @abstractmethod
    def flash(self) -> FlashResult:
        """将固件烧录至设备，返回结构化烧录结果。"""
        raise NotImplementedError

    @abstractmethod
    def inject_input(self, event: InputEvent) -> None:
        """向设备注入一次输入事件以触发被测场景（需求 2.6 / 19.2）。"""
        raise NotImplementedError

    @abstractmethod
    def send_cmd(self, cmd: str) -> str:
        """向设备发送命令并返回其响应（需求 14.4）。"""
        raise NotImplementedError


class HumanInLoopMixin:
    """把不可脚本化的操作转为返回 ``InterventionRequest`` 的可复用混入（需求 14.2）。

    半自动后端（如 ``HalfAutoDeviceIO``）可在无法自动执行 build/flash/inject 时
    复用这些实现：它们不抛异常，而是返回一个描述性的人在环干预请求，交由
    Half_Auto_Mode 在配置的干预点暂停等待人工完成。

    子类可只覆写其中可脚本化的方法，其余保留为干预点降级。
    """

    def request_intervention(
        self, action: str, instruction: str, **context: Any
    ) -> InterventionRequest:
        """构造一个人在环干预请求。"""
        return InterventionRequest(
            action=action, instruction=instruction, context=context
        )

    def build(self) -> InterventionRequest:  # type: ignore[override]
        """无法脚本化编译时，请求人工构建。"""
        return self.request_intervention(
            "build", "请手动执行编译（build），完成后在干预点确认以继续。"
        )

    def flash(self) -> InterventionRequest:  # type: ignore[override]
        """无法脚本化烧录时，请求人工烧录。"""
        return self.request_intervention(
            "flash", "请手动烧录固件（flash）至设备，完成后在干预点确认以继续。"
        )

    def inject_input(self, event: InputEvent) -> InterventionRequest:  # type: ignore[override]
        """无法脚本化注入时，请求人工触发输入。"""
        return self.request_intervention(
            "inject_input",
            f"请在设备上手动触发输入事件（kind={event.kind!r}），完成后确认以继续。",
            event=event.model_dump(),
        )
