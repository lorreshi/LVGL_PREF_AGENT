"""任务 4.2：SerialDeviceIO 全自动后端单元测试（需求 2.1, 2.3, 2.4, 14.4）。

全部用 mock，绝不接真机、绝不发起真实串口 / 子进程副作用：

* 波特率超限拒绝并报风险，且拒绝时**不尝试打开串口**（需求 2.3）；
* 串口打不开时抛出带**端口与原因**的描述性错误（需求 2.4）；
* ``send_cmd`` 往返：经串口写命令并返回解码后的响应（需求 14.4）；
* ``build`` / ``flash`` 执行配置命令（mock runner），成功/失败/未配置分支；
* ``capture`` 把串口字节流边读边落盘为 ``RawTraceArtifact``（需求 2.2）；
* import 触发注册，``DeviceIOFactory`` 依 ``type=="serial"`` 构造出 SerialDeviceIO。

真实串口连通性测试延后：需**显式 opt-in**（``RUN_REAL_SERIAL_TESTS=1`` 且提供
``DEVICE_SERIAL_PORT``），默认 skip、绝不 fail，参照 test_llm_providers 的做法。
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock

import pytest

from embedded_device_agent.core.config.models import DeviceConfig
from embedded_device_agent.core.device import DeviceIO, DeviceIOFactory
from embedded_device_agent.core.device.backends.serial import (
    BaudRateRiskError,
    SerialDeviceIO,
    SerialDeviceIOError,
    SerialOpenError,
)
from embedded_device_agent.core.device.models import BuildResult, FlashResult, InputEvent
from embedded_device_agent.core.models import RawTraceArtifact


def _cfg(**overrides) -> DeviceConfig:
    base = dict(
        type="serial",
        port="/dev/ttyUSB0",
        baud=115200,
        max_safe_baud=921600,
        build_cmd="make build",
        flash_cmd="make flash",
    )
    base.update(overrides)
    return DeviceConfig(**base)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> "subprocess.CompletedProcess[str]":
    return subprocess.CompletedProcess(
        args=["x"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# 需求 2.3：波特率超限拒绝并报风险，且不尝试打开串口
# ---------------------------------------------------------------------------
def test_open_serial_rejects_baud_over_max_safe_without_opening() -> None:
    factory = MagicMock()
    dev = SerialDeviceIO(_cfg(max_safe_baud=115200), serial_factory=factory)

    with pytest.raises(BaudRateRiskError) as exc:
        dev.open_serial("/dev/ttyUSB0", 921600)

    msg = str(exc.value)
    assert "921600" in msg  # 报出请求的波特率
    assert "115200" in msg  # 报出安全上限
    assert "/dev/ttyUSB0" in msg  # 报出端口
    factory.assert_not_called()  # 拒绝时绝不打开串口


def test_open_serial_allows_baud_at_limit() -> None:
    factory = MagicMock(return_value=MagicMock())
    dev = SerialDeviceIO(_cfg(max_safe_baud=115200), serial_factory=factory)
    dev.open_serial("/dev/ttyUSB0", 115200)  # 等于上限：允许
    factory.assert_called_once()


# ---------------------------------------------------------------------------
# 需求 2.4：串口打不开时抛出带端口与原因的描述性错误
# ---------------------------------------------------------------------------
def test_open_serial_failure_raises_descriptive_error_with_port_and_reason() -> None:
    def boom(**kwargs):
        raise OSError("device busy")

    dev = SerialDeviceIO(_cfg(), serial_factory=boom)
    with pytest.raises(SerialOpenError) as exc:
        dev.open_serial("/dev/ttyACM1", 115200)

    msg = str(exc.value)
    assert "/dev/ttyACM1" in msg  # 端口
    assert "device busy" in msg  # 原因


def test_open_serial_success_records_state_and_passes_args() -> None:
    port = MagicMock()
    factory = MagicMock(return_value=port)
    dev = SerialDeviceIO(_cfg(), serial_factory=factory, read_timeout_s=2.0)
    dev.open_serial("/dev/ttyUSB9", 460800)

    factory.assert_called_once_with(
        port="/dev/ttyUSB9", baudrate=460800, timeout=2.0
    )
    assert dev._serial is port
    assert dev._open_baud == 460800


# ---------------------------------------------------------------------------
# 需求 14.4：send_cmd 往返
# ---------------------------------------------------------------------------
def test_send_cmd_writes_command_and_returns_decoded_response() -> None:
    port = MagicMock()
    port.readline.return_value = b"pong\r\n"
    dev = SerialDeviceIO(_cfg(), serial_factory=MagicMock(return_value=port))
    dev.open_serial("/dev/ttyUSB0", 115200)

    resp = dev.send_cmd("ping")

    port.write.assert_called_once_with(b"ping\n")
    assert resp == "pong"  # 去除行尾并解码


def test_send_cmd_auto_opens_when_not_yet_open() -> None:
    port = MagicMock()
    port.readline.return_value = b"ok\n"
    factory = MagicMock(return_value=port)
    dev = SerialDeviceIO(_cfg(port="/dev/ttyUSB0", baud=115200), serial_factory=factory)

    assert dev.send_cmd("hi") == "ok"
    factory.assert_called_once()  # 自动打开


def test_send_cmd_without_port_config_raises() -> None:
    dev = SerialDeviceIO(_cfg(port=None), serial_factory=MagicMock())
    with pytest.raises(SerialDeviceIOError):
        dev.send_cmd("hi")


# ---------------------------------------------------------------------------
# 需求 2.2：capture 边读边落盘为 RawTraceArtifact
# ---------------------------------------------------------------------------
def test_capture_streams_to_disk_and_returns_artifact(tmp_path) -> None:
    port = MagicMock()
    # 读到若干块后持续返回空块（模拟无更多数据）；用生成器避免被快循环耗尽
    _chunks = iter([b"B|1|foo\n", b"E|1|foo\n"])
    port.read.side_effect = lambda *a, **k: next(_chunks, b"")
    dev = SerialDeviceIO(
        _cfg(),
        serial_factory=MagicMock(return_value=port),
        artifacts_dir=tmp_path,
    )
    dev.open_serial("/dev/ttyUSB0", 115200)

    artifact = dev.capture(duration_s=0.05)

    assert isinstance(artifact, RawTraceArtifact)
    assert artifact.path.exists()
    assert artifact.path.read_bytes() == b"B|1|foo\nE|1|foo\n"
    assert artifact.duration_s == 0.05
    assert artifact.baud == 115200
    assert artifact.run_id  # 非空标识


# ---------------------------------------------------------------------------
# build / flash：执行配置命令（mock runner）
# ---------------------------------------------------------------------------
def test_build_runs_command_and_reports_success() -> None:
    runner = MagicMock(return_value=_completed(0, stdout="compiled ok"))
    dev = SerialDeviceIO(_cfg(build_cmd="make -j build"), runner=runner)

    result = dev.build()

    runner.assert_called_once_with(["make", "-j", "build"])
    assert isinstance(result, BuildResult)
    assert result.success is True
    assert "compiled ok" in result.output


def test_build_reports_failure_reason_from_stderr() -> None:
    runner = MagicMock(return_value=_completed(2, stderr="error: undefined ref"))
    dev = SerialDeviceIO(_cfg(), runner=runner)

    result = dev.build()
    assert result.success is False
    assert "undefined ref" in (result.error or "")


def test_build_without_cmd_returns_failure_result() -> None:
    dev = SerialDeviceIO(_cfg(build_cmd=None))
    result = dev.build()
    assert result.success is False
    assert "build_cmd" in (result.error or "")


def test_flash_runs_command_and_reports_success() -> None:
    runner = MagicMock(return_value=_completed(0, stdout="flashed"))
    dev = SerialDeviceIO(_cfg(flash_cmd="make flash"), runner=runner)

    result = dev.flash()
    runner.assert_called_once_with(["make", "flash"])
    assert isinstance(result, FlashResult)
    assert result.success is True


def test_flash_without_cmd_returns_failure_result() -> None:
    dev = SerialDeviceIO(_cfg(flash_cmd=None))
    result = dev.flash()
    assert result.success is False
    assert "flash_cmd" in (result.error or "")


# ---------------------------------------------------------------------------
# inject_input：经串口写事件
# ---------------------------------------------------------------------------
def test_inject_input_writes_event_over_serial() -> None:
    port = MagicMock()
    dev = SerialDeviceIO(_cfg(), serial_factory=MagicMock(return_value=port))
    dev.open_serial("/dev/ttyUSB0", 115200)

    dev.inject_input(InputEvent(kind="touch", params={"x": 10, "y": 20}))

    assert port.write.call_count == 1
    written = port.write.call_args.args[0]
    assert b"touch" in written
    assert written.endswith(b"\n")


# ---------------------------------------------------------------------------
# 需求 12.4 / 2.1：import 触发注册，工厂依 type 分发构造 SerialDeviceIO
# ---------------------------------------------------------------------------
def test_factory_creates_serial_device_io() -> None:
    dev = DeviceIOFactory.create(_cfg())
    assert isinstance(dev, SerialDeviceIO)
    assert isinstance(dev, DeviceIO)


# ---------------------------------------------------------------------------
# 真实串口连通性测试（延后）：默认恒 skip，绝不 fail，也绝不接真机。
#
# 需**显式 opt-in** 才运行：同时满足
#   1) 设置 RUN_REAL_SERIAL_TESTS=1（显式开关，防止本地/CI 因恰好插着设备而误触发）；
#   2) 提供 DEVICE_SERIAL_PORT（真实串口设备路径）。
# 后续接真机验证时，导出 DEVICE_SERIAL_PORT 并设置 RUN_REAL_SERIAL_TESTS=1 即可。
# ---------------------------------------------------------------------------
_REAL_SERIAL_OPT_IN = os.environ.get("RUN_REAL_SERIAL_TESTS") == "1"


@pytest.mark.skipif(
    not (_REAL_SERIAL_OPT_IN and os.environ.get("DEVICE_SERIAL_PORT")),
    reason="真实串口连通性测试延后：需 RUN_REAL_SERIAL_TESTS=1 且 DEVICE_SERIAL_PORT",
)
def test_real_serial_open_and_capture() -> None:  # pragma: no cover - 需真机
    port = os.environ["DEVICE_SERIAL_PORT"]
    dev = SerialDeviceIO(_cfg(port=port, baud=115200))
    dev.open_serial(port, 115200)
    try:
        artifact = dev.capture(duration_s=1.0)
        assert artifact.path.exists()
    finally:
        dev.close()
