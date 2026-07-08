"""任务 4.3：HalfAutoDeviceIO 半自动后端单元测试。

覆盖三条主线（无需真机，用桩/mock）：

* **不可脚本化操作降级为人在环干预点而非抛错**（需求 14.2 / 7.3 / 11.3）：
  ``build`` / ``flash`` 在无对应命令配置时、``inject_input`` 在半自动模式下，
  返回 ``InterventionRequest`` 而非抛异常。
* **可脚本化操作正常委托给内部 SerialDeviceIO**（需求 2.1/2.2/14.4 复用）：
  ``open_serial`` / ``capture`` / ``send_cmd`` 委托给注入的串口实现；配置了
  ``build_cmd`` / ``flash_cmd`` 时 ``build`` / ``flash`` 也自动执行。
* **工厂按 ``type == "half_auto"`` 分发构造**（需求 12.4）。

真实串口连通性不在本单测覆盖范围（默认 skip、需显式 opt-in）。
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from embedded_device_agent.core.config.models import DeviceConfig
from embedded_device_agent.core.device import (
    DeviceIOFactory,
    HumanInLoopMixin,
    InterventionRequest,
)
from embedded_device_agent.core.device.backends.half_auto import HalfAutoDeviceIO
from embedded_device_agent.core.device.backends.serial import SerialDeviceIO
from embedded_device_agent.core.device.models import (
    BuildResult,
    FlashResult,
    InputEvent,
)


class _FakeSerialIO:
    """记录被委托调用的串口桩，实现 HalfAutoDeviceIO 所委托的可脚本化子集。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def open_serial(self, port: str, baud: int) -> None:
        self.calls.append(("open_serial", (port, baud)))

    def capture(self, duration_s: float) -> str:
        self.calls.append(("capture", (duration_s,)))
        return f"artifact::{duration_s}"

    def send_cmd(self, cmd: str) -> str:
        self.calls.append(("send_cmd", (cmd,)))
        return f"resp::{cmd}"

    def build(self) -> BuildResult:
        self.calls.append(("build", ()))
        return BuildResult(success=True, output="built")

    def flash(self) -> FlashResult:
        self.calls.append(("flash", ()))
        return FlashResult(success=True, output="flashed")

    def close(self) -> None:
        self.calls.append(("close", ()))


def _cfg(**overrides: Any) -> DeviceConfig:
    base: dict[str, Any] = {"type": "half_auto", "port": "/dev/ttyUSB0"}
    base.update(overrides)
    return DeviceConfig(**base)


# ---------------------------------------------------------------------------
# 可脚本化操作：委托给内部 SerialDeviceIO
# ---------------------------------------------------------------------------
def test_open_serial_delegates_to_serial() -> None:
    fake = _FakeSerialIO()
    dev = HalfAutoDeviceIO(_cfg(), serial_io=fake)  # type: ignore[arg-type]

    dev.open_serial("/dev/ttyUSB1", 115200)

    assert ("open_serial", ("/dev/ttyUSB1", 115200)) in fake.calls


def test_capture_delegates_to_serial() -> None:
    fake = _FakeSerialIO()
    dev = HalfAutoDeviceIO(_cfg(), serial_io=fake)  # type: ignore[arg-type]

    result = dev.capture(2.5)

    assert result == "artifact::2.5"
    assert ("capture", (2.5,)) in fake.calls


def test_send_cmd_delegates_to_serial() -> None:
    fake = _FakeSerialIO()
    dev = HalfAutoDeviceIO(_cfg(), serial_io=fake)  # type: ignore[arg-type]

    assert dev.send_cmd("reboot") == "resp::reboot"
    assert ("send_cmd", ("reboot",)) in fake.calls


def test_close_delegates_to_serial() -> None:
    fake = _FakeSerialIO()
    dev = HalfAutoDeviceIO(_cfg(), serial_io=fake)  # type: ignore[arg-type]

    dev.close()

    assert ("close", ()) in fake.calls


# ---------------------------------------------------------------------------
# 部分脚本化：配置了命令则自动执行
# ---------------------------------------------------------------------------
def test_build_delegates_when_build_cmd_configured() -> None:
    fake = _FakeSerialIO()
    dev = HalfAutoDeviceIO(_cfg(build_cmd="make"), serial_io=fake)  # type: ignore[arg-type]

    result = dev.build()

    assert isinstance(result, BuildResult)
    assert result.success is True
    assert ("build", ()) in fake.calls


def test_flash_delegates_when_flash_cmd_configured() -> None:
    fake = _FakeSerialIO()
    dev = HalfAutoDeviceIO(_cfg(flash_cmd="flash.sh"), serial_io=fake)  # type: ignore[arg-type]

    result = dev.flash()

    assert isinstance(result, FlashResult)
    assert result.success is True
    assert ("flash", ()) in fake.calls


# ---------------------------------------------------------------------------
# 不可脚本化操作：降级为人在环干预点（返回 InterventionRequest，不抛错）
# ---------------------------------------------------------------------------
def test_build_returns_intervention_when_no_build_cmd() -> None:
    fake = _FakeSerialIO()
    dev = HalfAutoDeviceIO(_cfg(), serial_io=fake)  # type: ignore[arg-type]

    result = dev.build()

    assert isinstance(result, InterventionRequest)
    assert result.action == "build"
    assert result.instruction  # 非空可读指引
    # 未触发底层自动构建
    assert ("build", ()) not in fake.calls


def test_flash_returns_intervention_when_no_flash_cmd() -> None:
    fake = _FakeSerialIO()
    dev = HalfAutoDeviceIO(_cfg(), serial_io=fake)  # type: ignore[arg-type]

    result = dev.flash()

    assert isinstance(result, InterventionRequest)
    assert result.action == "flash"
    assert result.instruction
    assert ("flash", ()) not in fake.calls


def test_inject_input_returns_intervention() -> None:
    fake = _FakeSerialIO()
    dev = HalfAutoDeviceIO(_cfg(), serial_io=fake)  # type: ignore[arg-type]

    event = InputEvent(kind="touch", params={"x": 10, "y": 20})
    result = dev.inject_input(event)

    assert isinstance(result, InterventionRequest)
    assert result.action == "inject_input"
    # 事件信息进入 context，便于操作者复现
    assert result.context.get("event", {}).get("kind") == "touch"


def test_intervention_never_raises() -> None:
    """人在环降级的关键契约：不抛错（需求 14.2）。"""
    dev = HalfAutoDeviceIO(_cfg())

    # 无 build_cmd/flash_cmd 且未打开真实串口，这些调用也不得抛异常
    assert isinstance(dev.build(), InterventionRequest)
    assert isinstance(dev.flash(), InterventionRequest)
    assert isinstance(dev.inject_input(InputEvent(kind="key")), InterventionRequest)


def test_is_human_in_loop_and_device_io() -> None:
    dev = HalfAutoDeviceIO(_cfg())
    from embedded_device_agent.core.device.base import DeviceIO

    assert isinstance(dev, HumanInLoopMixin)
    assert isinstance(dev, DeviceIO)


# ---------------------------------------------------------------------------
# 工厂分发：type == "half_auto"
# ---------------------------------------------------------------------------
def test_factory_dispatches_half_auto() -> None:
    dev = DeviceIOFactory.create(_cfg())

    assert isinstance(dev, HalfAutoDeviceIO)


def test_factory_default_serial_io_is_serial_backend() -> None:
    """未注入 serial_io 时，内部应组合一个真实的 SerialDeviceIO。"""
    dev = HalfAutoDeviceIO(_cfg())

    assert isinstance(dev._serial, SerialDeviceIO)


# ---------------------------------------------------------------------------
# 真实串口连通性：默认 skip，仅在显式 opt-in 时运行
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("EDA_REAL_SERIAL_PORT") is None,
    reason="需设置 EDA_REAL_SERIAL_PORT 显式 opt-in 才运行真实串口测试",
)
def test_real_serial_roundtrip() -> None:  # pragma: no cover - 需真机
    port = os.environ["EDA_REAL_SERIAL_PORT"]
    dev = HalfAutoDeviceIO(_cfg(port=port))
    dev.open_serial(port, 115200)
    try:
        assert isinstance(dev.send_cmd("ping"), str)
    finally:
        dev.close()
