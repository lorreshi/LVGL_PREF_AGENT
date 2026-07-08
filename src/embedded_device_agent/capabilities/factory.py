"""能力工厂 ``CapabilityFactory``（任务 14.1）。

严格对照 design.md「平台层 → BaseCapability 与 CapabilityFactory」与需求 16.2/16.4：
基于字符串键的**装饰器注册表**与 ``create(key, core)`` 构造，使任一遵循
``BaseCapability`` 契约的能力经 ``@register("...")`` 注册后即可被平台发现与构造，
**无需改动路由与底座**（需求 16.3，Property 10）。

用法::

    @CapabilityFactory.register("perf_tuning")
    class PerfTuningCapability(BaseCapability):
        ...

    cap = CapabilityFactory.create("perf_tuning", core)

``available()`` 返回全部已注册能力键，供 Router 枚举分发候选（需求 17.2）。
"""

from __future__ import annotations

from typing import Callable

from embedded_device_agent.capabilities.base import BaseCapability
from embedded_device_agent.core.services import CoreServices

__all__ = ["CapabilityFactory"]


class CapabilityFactory:
    """依字符串键分发构造 ``BaseCapability`` 具体实现的工厂（需求 16.2/16.4）。"""

    _registry: dict[str, type[BaseCapability]] = {}

    @classmethod
    def register(
        cls, key: str
    ) -> Callable[[type[BaseCapability]], type[BaseCapability]]:
        """装饰器：将某个 ``BaseCapability`` 子类以 ``key`` 注册到工厂。

        重复注册同一 ``key`` 视为配置错误，直接报错以避免静默覆盖。
        """

        def decorator(capability_cls: type[BaseCapability]) -> type[BaseCapability]:
            if not isinstance(capability_cls, type) or not issubclass(
                capability_cls, BaseCapability
            ):
                raise TypeError(
                    f"注册到 CapabilityFactory 的类型必须是 BaseCapability 的子类，"
                    f"但收到：{capability_cls!r}"
                )
            if key in cls._registry:
                raise ValueError(
                    f"capability key '{key}' 已注册为 "
                    f"{cls._registry[key].__name__}，不允许重复注册。"
                )
            cls._registry[key] = capability_cls
            return capability_cls

        return decorator

    @classmethod
    def create(cls, key: str, core: CoreServices) -> BaseCapability:
        """依据 ``key`` 构造对应的 ``BaseCapability`` 实例。

        未知 ``key`` 时报出清晰错误并列出已注册能力，便于定位配置问题。
        ``core`` 透传给能力构造，保留其在构造期即持有共享底座的可能。
        """
        capability_cls = cls._registry.get(key)
        if capability_cls is None:
            available = ", ".join(sorted(cls._registry)) or "(无)"
            raise ValueError(
                f"未知的 capability key '{key}'。已注册的可用能力：{available}。"
            )
        return capability_cls()

    @classmethod
    def available(cls) -> list[str]:
        """返回全部已注册能力键（有序），供 Router 枚举分发候选（需求 17.2）。"""
        return sorted(cls._registry)
