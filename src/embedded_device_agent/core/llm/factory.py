"""LLM_Provider 工厂。

严格对照 design.md "LLM_Provider" 章节：基于 ``LLMConfig.type`` 字段的**装饰器
注册表**与 ``create(cfg)`` 构造，满足需求 12.3——请求某个 LLM_Provider 时，工厂
从 ``BaseLLMProvider`` 基类构造 YAML 配置中命名（``type``）的提供方。

用法::

    @LLMProviderFactory.register("anthropic")
    class AnthropicProvider(BaseLLMProvider):
        ...

    provider = LLMProviderFactory.create(cfg)   # cfg.type == "anthropic"

新增后端零改动工厂：只需在其模块中用 ``@register`` 装饰并确保该模块被导入。
"""

from __future__ import annotations

from typing import Callable

from embedded_device_agent.core.config.models import LLMConfig
from embedded_device_agent.core.llm.base import BaseLLMProvider

__all__ = ["LLMProviderFactory"]


class LLMProviderFactory:
    """依 ``LLMConfig.type`` 分发构造 ``BaseLLMProvider`` 具体实现的工厂。"""

    _registry: dict[str, type[BaseLLMProvider]] = {}

    @classmethod
    def register(
        cls, key: str
    ) -> Callable[[type[BaseLLMProvider]], type[BaseLLMProvider]]:
        """装饰器：将某个 ``BaseLLMProvider`` 子类以 ``key`` 注册到工厂。

        重复注册同一 ``key`` 视为配置错误，直接报错以避免静默覆盖。
        """

        def decorator(provider_cls: type[BaseLLMProvider]) -> type[BaseLLMProvider]:
            if not isinstance(provider_cls, type) or not issubclass(
                provider_cls, BaseLLMProvider
            ):
                raise TypeError(
                    f"注册到 LLMProviderFactory 的类型必须是 BaseLLMProvider 的子类，"
                    f"但收到：{provider_cls!r}"
                )
            if key in cls._registry:
                raise ValueError(
                    f"LLM provider type '{key}' 已注册为 "
                    f"{cls._registry[key].__name__}，不允许重复注册。"
                )
            cls._registry[key] = provider_cls
            return provider_cls

        return decorator

    @classmethod
    def create(cls, cfg: LLMConfig) -> BaseLLMProvider:
        """依据 ``cfg.type`` 构造并返回对应的 ``BaseLLMProvider`` 实例。

        未知 ``type`` 时报出清晰错误，列出当前已注册的可用类型，便于定位
        配置问题（需求 12.2 的精神：在运行前给出具体原因）。
        """
        provider_cls = cls._registry.get(cfg.type)
        if provider_cls is None:
            available = ", ".join(sorted(cls._registry)) or "(无)"
            raise ValueError(
                f"未知的 LLM provider type '{cfg.type}'。"
                f"已注册的可用类型：{available}。"
            )
        return provider_cls(cfg)
