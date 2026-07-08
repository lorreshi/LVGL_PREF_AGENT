"""Retriever 工厂。

严格对照 design.md "Retriever" 章节：基于 ``RetrieverConfig.type`` 字段的
**装饰器注册表**与 ``create(cfg)`` 构造，满足需求 12.5——请求某个 Retriever
后端时，工厂从 ``BaseRetriever`` 基类构造 YAML 配置中命名（``type``）的实现
（local_store / external_rag）。

风格与 ``LLMProviderFactory`` / ``DeviceIOFactory``（任务 3.1 / 4.1）保持一致。
用法::

    @RetrieverFactory.register("local_store")
    class LocalStoreRetriever(BaseRetriever):
        ...

    retriever = RetrieverFactory.create(cfg)   # cfg.type == "local_store"

新增后端零改动工厂：只需在其模块中用 ``@register`` 装饰并确保该模块被导入。
"""

from __future__ import annotations

from typing import Callable

from embedded_device_agent.core.config.models import RetrieverConfig
from embedded_device_agent.core.memory.base import BaseRetriever

__all__ = ["RetrieverFactory"]


class RetrieverFactory:
    """依 ``RetrieverConfig.type`` 分发构造 ``BaseRetriever`` 具体实现的工厂。"""

    _registry: dict[str, type[BaseRetriever]] = {}

    @classmethod
    def register(
        cls, key: str
    ) -> Callable[[type[BaseRetriever]], type[BaseRetriever]]:
        """装饰器：将某个 ``BaseRetriever`` 子类以 ``key`` 注册到工厂。

        重复注册同一 ``key`` 视为配置错误，直接报错以避免静默覆盖。
        """

        def decorator(retriever_cls: type[BaseRetriever]) -> type[BaseRetriever]:
            if not isinstance(retriever_cls, type) or not issubclass(
                retriever_cls, BaseRetriever
            ):
                raise TypeError(
                    f"注册到 RetrieverFactory 的类型必须是 BaseRetriever 的子类，"
                    f"但收到：{retriever_cls!r}"
                )
            if key in cls._registry:
                raise ValueError(
                    f"Retriever type '{key}' 已注册为 "
                    f"{cls._registry[key].__name__}，不允许重复注册。"
                )
            cls._registry[key] = retriever_cls
            return retriever_cls

        return decorator

    @classmethod
    def create(cls, cfg: RetrieverConfig) -> BaseRetriever:
        """依据 ``cfg.type`` 构造并返回对应的 ``BaseRetriever`` 实例。

        未知 ``type`` 时报出清晰错误，列出当前已注册的可用类型，便于定位
        配置问题（需求 12.2 的精神：在运行前给出具体原因）。
        """
        retriever_cls = cls._registry.get(cfg.type)
        if retriever_cls is None:
            available = ", ".join(sorted(cls._registry)) or "(无)"
            raise ValueError(
                f"未知的 retriever type '{cfg.type}'。"
                f"已注册的可用类型：{available}。"
            )
        return retriever_cls(cfg)
