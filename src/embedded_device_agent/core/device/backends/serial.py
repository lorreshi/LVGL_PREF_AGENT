"""全自动 SerialDeviceIO 后端（任务 4.2）。

用 pyserial 实现 ``DeviceIO`` 契约的全自动实现（design.md "DeviceIO →
``SerialDeviceIO``：pyserial 实现，全自动"）。所有真实串口 / 子进程副作用都藏在
本类内部，经 ``@DeviceIOFactory.register("serial")`` 注册，由 YAML 的
``device.type == "serial"`` 选择（需求 12.4）。

关键行为（严格对照需求）：

* ``open_serial(port, baud)``：打开前校验 ``baud <= max_safe_baud``，超限拒绝并
  报风险（需求 2.3）；串口打不开时抛出带**端口与原因**的描述性错误（需求 2.4）。
* ``capture(duration_s)``：在给定时长内把串口字节流**边读边落盘**为原始日志文件，
  返回指向该文件的 ``RawTraceArtifact``；日志内容不在内存中长期累积（Property 12
  的精神：大日志留在磁盘）。
* ``build()`` / ``flash()``：执行 ``DeviceConfig.build_cmd`` / ``flash_cmd``
  （可脚本化时），返回结构化 ``BuildResult`` / ``FlashResult``（需求 6.3 / 7.2）。
* ``send_cmd(cmd)``：经已打开串口发命令并返回设备响应（需求 14.4）。
* ``inject_input(event)``：经串口注入一次输入事件以触发被测场景（需求 2.5 / 2.6）。

依赖注入友好：``serial_factory`` 与 ``runner`` 可在测试中替换，从而无需真实硬件
即可对全部路径做单元测试（真实串口连通性测试见 tests，默认 skip、需显式 opt-in）。
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import serial  # pyserial

from embedded_device_agent.core.config.models import DeviceConfig
from embedded_device_agent.core.device.base import DeviceIO
from embedded_device_agent.core.device.factory import DeviceIOFactory
from embedded_device_agent.core.device.models import (
    BuildResult,
    FlashResult,
    InputEvent,
)
from embedded_device_agent.core.models import RawTraceArtifact

__all__ = [
    "SerialDeviceIO",
    "SerialDeviceIOError",
    "BaudRateRiskError",
    "SerialOpenError",
]


# ---------------------------------------------------------------------------
# 描述性错误类型（携带端口与原因，满足需求 2.3 / 2.4 的"报风险 / 报原因"）
# ---------------------------------------------------------------------------
class SerialDeviceIOError(RuntimeError):
    """SerialDeviceIO 相关错误的基类。"""


class BaudRateRiskError(SerialDeviceIOError):
    """请求的波特率超过配置声明的安全上限——拒绝打开并报风险（需求 2.3）。"""


class SerialOpenError(SerialDeviceIOError):
    """串口无法打开——携带端口与失败原因的描述性错误（需求 2.4）。"""


class _SerialPort(Protocol):
    """本后端实际用到的 pyserial ``Serial`` 子集接口（便于测试替换）。"""

    def read(self, size: int = ...) -> bytes: ...
    def readline(self) -> bytes: ...
    def write(self, data: bytes) -> int: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


# 打开串口的工厂：默认用 pyserial 的 Serial；测试可注入桩。
SerialFactory = Callable[..., _SerialPort]
# 执行外部命令（build/flash）：默认用 subprocess.run；测试可注入桩。
CommandRunner = Callable[..., "subprocess.CompletedProcess[str]"]


def _default_serial_factory(**kwargs: Any) -> _SerialPort:
    return serial.Serial(**kwargs)


def _default_runner(args: list[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(args, capture_output=True, text=True)


@DeviceIOFactory.register("serial")
class SerialDeviceIO(DeviceIO):
    """基于 pyserial 的全自动设备控制面实现。

    工厂经 ``SerialDeviceIO(cfg)`` 构造（见 ``DeviceIOFactory.create``），故除
    ``cfg`` 外的参数均为可选，仅供测试注入桩使用。
    """

    def __init__(
        self,
        cfg: DeviceConfig,
        *,
        artifacts_dir: str | Path | None = None,
        serial_factory: SerialFactory | None = None,
        runner: CommandRunner | None = None,
        read_timeout_s: float = 1.0,
        read_chunk_bytes: int = 4096,
        line_terminator: str = "\n",
        output_tail_chars: int = 4000,
    ) -> None:
        self.cfg = cfg
        self._artifacts_dir = (
            Path(artifacts_dir)
            if artifacts_dir is not None
            else Path.cwd() / ".artifacts" / "raw_traces"
        )
        self._serial_factory: SerialFactory = serial_factory or _default_serial_factory
        self._runner: CommandRunner = runner or _default_runner
        self._read_timeout_s = read_timeout_s
        self._read_chunk_bytes = read_chunk_bytes
        self._line_terminator = line_terminator
        self._output_tail_chars = output_tail_chars

        self._serial: _SerialPort | None = None
        self._open_port: str | None = None
        self._open_baud: int | None = None

    # -- 串口开合 -----------------------------------------------------------
    def open_serial(self, port: str, baud: int) -> None:
        """打开串口连接（需求 2.1）。

        打开前先校验波特率安全性（需求 2.3），再尝试打开；打不开抛出带端口与
        原因的描述性错误（需求 2.4）。
        """
        max_safe = self.cfg.max_safe_baud
        if baud > max_safe:
            raise BaudRateRiskError(
                f"拒绝打开串口 {port!r}：请求波特率 {baud} 超过配置声明的安全上限 "
                f"max_safe_baud={max_safe}，存在丢帧/误码风险。请降低波特率或调整配置。"
            )
        try:
            self._serial = self._serial_factory(
                port=port, baudrate=baud, timeout=self._read_timeout_s
            )
        except Exception as exc:  # pyserial 抛 serial.SerialException 等
            raise SerialOpenError(
                f"无法打开串口 {port!r}（波特率 {baud}）：{exc}"
            ) from exc
        self._open_port = port
        self._open_baud = baud

    def _ensure_open(self) -> _SerialPort:
        """确保串口已打开；未打开则依据 ``cfg`` 自动打开。"""
        if self._serial is None:
            if not self.cfg.port:
                raise SerialDeviceIOError(
                    "串口未打开且配置未提供 device.port，无法自动打开串口。"
                )
            self.open_serial(self.cfg.port, self.cfg.baud)
        assert self._serial is not None
        return self._serial

    def close(self) -> None:
        """关闭已打开的串口（若有）。"""
        if self._serial is not None:
            self._serial.close()
            self._serial = None
            self._open_port = None
            self._open_baud = None

    # -- 采集 ---------------------------------------------------------------
    def capture(self, duration_s: float) -> RawTraceArtifact:
        """采集 ``duration_s`` 秒串口输出，边读边落盘，返回原始日志工件（需求 2.2）。

        字节流以块为单位直接写入磁盘文件，不在内存中长期累积（大日志留在磁盘，
        state 中只传 ``RawTraceArtifact`` 引用）。
        """
        port = self._ensure_open()
        run_id = uuid.uuid4().hex
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self._artifacts_dir / f"{run_id}_raw.log"

        deadline = time.monotonic() + max(0.0, duration_s)
        with path.open("wb") as fh:
            while time.monotonic() < deadline:
                chunk = port.read(self._read_chunk_bytes)
                if chunk:
                    fh.write(chunk)

        return RawTraceArtifact(
            run_id=run_id,
            path=path,
            captured_at=datetime.now(timezone.utc),
            duration_s=duration_s,
            baud=self._open_baud if self._open_baud is not None else self.cfg.baud,
        )

    # -- 构建 / 烧录 --------------------------------------------------------
    def build(self) -> BuildResult:
        """执行 ``cfg.build_cmd`` 触发固件编译（需求 6.3）。"""
        if not self.cfg.build_cmd:
            return BuildResult(
                success=False,
                error="未配置 device.build_cmd，SerialDeviceIO 无法自动编译固件。",
            )
        proc = self._runner(shlex.split(self.cfg.build_cmd))
        ok = proc.returncode == 0
        return BuildResult(
            success=ok,
            output=self._tail(proc.stdout),
            artifact_path=None,
            error=None if ok else self._failure_reason(proc, "build"),
        )

    def flash(self) -> FlashResult:
        """执行 ``cfg.flash_cmd`` 将固件烧录至设备（需求 6.3 / 7.2）。"""
        if not self.cfg.flash_cmd:
            return FlashResult(
                success=False,
                error="未配置 device.flash_cmd，SerialDeviceIO 无法自动烧录固件。",
            )
        proc = self._runner(shlex.split(self.cfg.flash_cmd))
        ok = proc.returncode == 0
        return FlashResult(
            success=ok,
            output=self._tail(proc.stdout),
            error=None if ok else self._failure_reason(proc, "flash"),
        )

    # -- 命令 / 输入 --------------------------------------------------------
    def send_cmd(self, cmd: str) -> str:
        """经已打开串口发送命令并返回设备响应（需求 14.4）。"""
        port = self._ensure_open()
        port.write((cmd + self._line_terminator).encode("utf-8"))
        port.flush()
        raw = port.readline()
        return raw.decode("utf-8", errors="replace").strip()

    def inject_input(self, event: InputEvent) -> None:
        """经串口注入一次输入事件以触发被测场景（需求 2.5 / 2.6）。"""
        port = self._ensure_open()
        payload = json.dumps(
            {"kind": event.kind, "params": event.params}, ensure_ascii=False
        )
        port.write((payload + self._line_terminator).encode("utf-8"))
        port.flush()

    # -- 内部辅助 -----------------------------------------------------------
    def _tail(self, text: str | None) -> str:
        if not text:
            return ""
        return text[-self._output_tail_chars :]

    def _failure_reason(
        self, proc: "subprocess.CompletedProcess[str]", action: str
    ) -> str:
        stderr = (proc.stderr or "").strip()
        if stderr:
            return self._tail(stderr)
        return f"{action} 命令以非零退出码 {proc.returncode} 结束。"
