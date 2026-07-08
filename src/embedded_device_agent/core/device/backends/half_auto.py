"""半自动 HalfAutoDeviceIO 后端（任务 4.3）。

对照 design.md "DeviceIO → ``HalfAutoDeviceIO``：可脚本化的部分自动化，其余操作
产出人在环干预点"，以及需求 14.2 / 7.3 / 11.3：

* **可脚本化的部分复用串口能力**：``open_serial`` / ``capture`` / ``send_cmd``
  这些纯串口往返操作与全自动后端并无差别，故本类**组合注入**一个
  ``SerialDeviceIO`` 并把这些调用直接委托给它——不重复实现串口副作用，保持
  "同一套设备控制面被复用"（需求 14.1 / 18.2）。
* **无法脚本化的操作降级为人在环干预点**：当 ``build`` / ``flash`` 在配置中
  未提供对应命令（``build_cmd`` / ``flash_cmd`` 缺省），或输入注入需人工在设备上
  触发时，本类**不抛错**，而是经 ``HumanInLoopMixin`` 返回一个描述性的
  ``InterventionRequest``（需求 14.2）。这与 Half_Auto_Mode 在
  ``collect``（人工触发场景）/ ``instrument``/``optimize``（人工烧录/批准）干预点
  暂停等待人工完成的语义一致（需求 7.3 / 11.3）。
* **部分脚本化**：若 YAML 为 ``build_cmd`` / ``flash_cmd`` 提供了可执行命令，
  则这些步骤仍可自动执行（委托给内部 ``SerialDeviceIO`` 运行命令），仅在
  确无自动化条件时才降级为干预点——即"可脚本化的部分自动化"。

经 ``@DeviceIOFactory.register("half_auto")`` 注册，由 YAML 的
``device.type == "half_auto"`` 选择（需求 12.4）；导入本模块即触发注册。

依赖注入友好：可注入自定义的 ``serial_io``（或透传串口构造参数），从而无需真实
硬件即可对全部路径做单元测试。
"""

from __future__ import annotations

from pathlib import Path

from embedded_device_agent.core.config.models import DeviceConfig
from embedded_device_agent.core.device.base import DeviceIO, HumanInLoopMixin
from embedded_device_agent.core.device.backends.serial import (
    CommandRunner,
    SerialDeviceIO,
    SerialFactory,
)
from embedded_device_agent.core.device.factory import DeviceIOFactory
from embedded_device_agent.core.device.models import (
    BuildResult,
    FlashResult,
    InputEvent,
    InterventionRequest,
)
from embedded_device_agent.core.models import RawTraceArtifact

__all__ = ["HalfAutoDeviceIO"]


@DeviceIOFactory.register("half_auto")
class HalfAutoDeviceIO(HumanInLoopMixin, DeviceIO):
    """部分脚本化 + 人在环的设备控制面实现（需求 14.2 / 7.3 / 11.3）。

    可脚本化的串口往返（``open_serial`` / ``capture`` / ``send_cmd``）委托给内部
    组合的 ``SerialDeviceIO``；``build`` / ``flash`` 在配置了对应命令时自动执行、
    否则降级为人在环干预点；``inject_input`` 在半自动模式下默认由人工在设备上触发，
    故降级为干预点（保留 ``HumanInLoopMixin`` 的实现）。

    工厂经 ``HalfAutoDeviceIO(cfg)`` 构造（见 ``DeviceIOFactory.create``），故除
    ``cfg`` 外的参数均为可选：可直接注入一个 ``serial_io`` 桩，或透传串口构造参数
    以便测试无需真实硬件。
    """

    def __init__(
        self,
        cfg: DeviceConfig,
        *,
        serial_io: SerialDeviceIO | None = None,
        artifacts_dir: str | Path | None = None,
        serial_factory: SerialFactory | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self.cfg = cfg
        # 组合复用：可脚本化的串口能力交由内部 SerialDeviceIO 承担，不重复实现。
        self._serial: SerialDeviceIO = serial_io or SerialDeviceIO(
            cfg,
            artifacts_dir=artifacts_dir,
            serial_factory=serial_factory,
            runner=runner,
        )

    # -- 可脚本化：委托给内部 SerialDeviceIO --------------------------------
    def open_serial(self, port: str, baud: int) -> None:
        """打开串口连接（复用串口能力，需求 2.1 / 2.3 / 2.4）。"""
        self._serial.open_serial(port, baud)

    def capture(self, duration_s: float) -> RawTraceArtifact:
        """采集原始 trace，边读边落盘并返回工件引用（复用串口能力，需求 2.2）。"""
        return self._serial.capture(duration_s)

    def send_cmd(self, cmd: str) -> str:
        """经已打开串口发送命令并返回设备响应（复用串口能力，需求 14.4）。"""
        return self._serial.send_cmd(cmd)

    def close(self) -> None:
        """关闭底层串口（若有）。"""
        self._serial.close()

    # -- 部分脚本化 / 人在环降级 --------------------------------------------
    def build(self) -> BuildResult | InterventionRequest:  # type: ignore[override]
        """配置了 ``build_cmd`` 时自动编译；否则请求人工构建（需求 7.3 / 14.2）。"""
        if self.cfg.build_cmd:
            return self._serial.build()
        return self.request_intervention(
            "build",
            "当前配置未提供可脚本化的 build 命令（device.build_cmd 缺省）。"
            "请手动执行编译（build），完成后在干预点确认以继续。",
        )

    def flash(self) -> FlashResult | InterventionRequest:  # type: ignore[override]
        """配置了 ``flash_cmd`` 时自动烧录；否则请求人工烧录（需求 7.3 / 14.2）。"""
        if self.cfg.flash_cmd:
            return self._serial.flash()
        return self.request_intervention(
            "flash",
            "当前配置未提供可脚本化的 flash 命令（device.flash_cmd 缺省）。"
            "请手动烧录固件（flash）至设备，完成后在干预点确认以继续。",
        )

    def inject_input(self, event: InputEvent) -> InterventionRequest:  # type: ignore[override]
        """半自动模式下由人工在设备上触发输入，降级为人在环干预点（需求 11.3 / 14.2）。

        直接复用 ``HumanInLoopMixin.inject_input`` 的实现（返回 ``InterventionRequest``
        而非抛错），语义与 Half_Auto_Mode 在 ``collect`` 处人工触发场景一致。
        """
        return HumanInLoopMixin.inject_input(self, event)
