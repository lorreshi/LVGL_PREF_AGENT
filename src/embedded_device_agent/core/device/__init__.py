"""DeviceIO 底座：DeviceIO 契约 + HumanInLoopMixin + DeviceIOFactory 装饰器注册工厂。

DeviceIO 是所有能力复用的设备控制面副作用边界（需求 14 / 18.2）。具体后端
（serial / half_auto / fake）由 ``backends/`` 子包实现并经
``@DeviceIOFactory.register(...)`` 注册（任务 4.2 / 4.3 及测试 harness）。
"""

from embedded_device_agent.core.device.base import DeviceIO, HumanInLoopMixin
from embedded_device_agent.core.device.factory import DeviceIOFactory
from embedded_device_agent.core.device.models import (
    BuildResult,
    FlashResult,
    InputEvent,
    InterventionRequest,
)

# 导入具体后端子包以触发 ``@DeviceIOFactory.register(...)`` 注册（任务 4.2 起）。
# 放在契约/工厂导入之后以避免循环导入。
from embedded_device_agent.core.device import backends as backends  # noqa: E402,F401

__all__ = [
    "DeviceIO",
    "HumanInLoopMixin",
    "DeviceIOFactory",
    "BuildResult",
    "FlashResult",
    "InputEvent",
    "InterventionRequest",
]
