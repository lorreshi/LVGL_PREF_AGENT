"""可插拔能力（Capability）模块的命名空间。

暴露平台契约 ``BaseCapability`` 与 ``CapabilityFactory``，并导入各具体能力模块以触发
其 ``@CapabilityFactory.register(...)`` 注册（Property 10：注册即可路由）。新增能力
只需在此追加一行导入即随平台加载自动注册。
"""

from __future__ import annotations

from embedded_device_agent.capabilities.base import BaseCapability
from embedded_device_agent.capabilities.factory import CapabilityFactory

# 导入具体能力模块以触发注册（放在契约/工厂之后以避免循环导入）。
from embedded_device_agent.capabilities.perf_tuning import capability as _perf_tuning  # noqa: E402,F401
from embedded_device_agent.capabilities.functional_test import (  # noqa: E402,F401
    capability as _functional_test,
)

__all__ = ["BaseCapability", "CapabilityFactory"]
