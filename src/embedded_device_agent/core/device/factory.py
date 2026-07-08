"""DeviceIO 工厂。

严格对照 design.md "DeviceIO" 章节：基于 ``DeviceConfig.type`` 字段的**装饰器
注册表**与 ``create(cfg)`` 构造，满足需求 12.4——请求某个设备后端时，工厂从
``DeviceIO`` 基类构造 YAML 配置中命名（``type``）的实现（serial / half_auto / fake）。

风格与 ``LLMProviderFactory``（任务 3.1）保持一致。用法::

    @DeviceIOFactory.register("serial")
    class SerialDeviceIO(DeviceIO):
        ...

    device = DeviceIOFactory.create(cfg)   # cfg.type == "serial"

新增后端零改动工厂：只需在其模块中用 ``@register`` 装饰并确保该模块被导入。
"""

from __future__ import annotations

from typing import Callable

from embedded_device_agent.core.config.models import DeviceConfig
from embedded_device_agent.core.device.base import DeviceIO

__all__ = ["DeviceIOFactory"]


class DeviceIOFactory:
    """依 ``DeviceConfig.type`` 分发构造 ``DeviceIO`` 具体实现的工厂。"""

    _registry: dict[str, type[DeviceIO]] = {}

    @classmethod
    def register(cls, key: str) -> Callable[[type[DeviceIO]], type[DeviceIO]]:
        """装饰器：将某个 ``DeviceIO`` 子类以 ``key`` 注册到工厂。

        重复注册同一 ``key`` 视为配置错误，直接报错以避免静默覆盖。
        """

        def decorator(device_cls: type[DeviceIO]) -> type[DeviceIO]:
            if not isinstance(device_cls, type) or not issubclass(device_cls, DeviceIO):
                raise TypeError(
                    f"注册到 DeviceIOFactory 的类型必须是 DeviceIO 的子类，"
                    f"但收到：{device_cls!r}"
                )
            if key in cls._registry:
                raise ValueError(
                    f"Device type '{key}' 已注册为 "
                    f"{cls._registry[key].__name__}，不允许重复注册。"
                )
            cls._registry[key] = device_cls
            return device_cls

        return decorator

    @classmethod
    def create(cls, cfg: DeviceConfig) -> DeviceIO:
        """依据 ``cfg.type`` 构造并返回对应的 ``DeviceIO`` 实例。

        未知 ``type`` 时报出清晰错误，列出当前已注册的可用类型，便于定位
        配置问题（需求 12.2 的精神：在运行前给出具体原因）。
        """
        device_cls = cls._registry.get(cfg.type)
        if device_cls is None:
            available = ", ".join(sorted(cls._registry)) or "(无)"
            raise ValueError(
                f"未知的 device type '{cfg.type}'。"
                f"已注册的可用类型：{available}。"
            )
        return device_cls(cfg)
