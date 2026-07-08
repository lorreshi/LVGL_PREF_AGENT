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

__all__ = [
    "DeviceIO",
    "HumanInLoopMixin",
    "DeviceIOFactory",
    "BuildResult",
    "FlashResult",
    "InputEvent",
    "InterventionRequest",
]
