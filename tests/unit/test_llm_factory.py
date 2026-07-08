"""任务 3.1：LLM_Provider 底座（BaseLLMProvider + LLMProviderFactory）单元测试。

验证工厂机制本身，不触达真实 LLM：
* 装饰器注册使子类进入注册表；
* ``create(cfg)`` 依 ``cfg.type`` 分发构造正确子类，并把 cfg 透传给构造函数；
* 未知 ``type`` 报出清晰错误（列出可用类型）；
* 非 BaseLLMProvider 子类注册被拒绝；
* 重复注册同一 key 被拒绝。

用一个 DummyProvider 注册来测工厂机制（无需真实模型后端）。
_Requirements: 12.3_
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from embedded_device_agent.core.config.models import LLMConfig
from embedded_device_agent.core.llm import BaseLLMProvider, LLMProviderFactory


@pytest.fixture
def isolated_registry():
    """隔离工厂注册表：每个用例前后清空并还原，避免用例间串扰。"""
    saved = dict(LLMProviderFactory._registry)
    LLMProviderFactory._registry.clear()
    try:
        yield LLMProviderFactory
    finally:
        LLMProviderFactory._registry.clear()
        LLMProviderFactory._registry.update(saved)


def _make_cfg(type_: str) -> LLMConfig:
    return LLMConfig(type=type_, model="dummy-model", api_key_env="DUMMY_KEY")


def test_register_decorator_adds_to_registry(isolated_registry):
    @isolated_registry.register("dummy")
    class DummyProvider(BaseLLMProvider):
        def __init__(self, cfg: LLMConfig) -> None:
            self.cfg = cfg

        def get_chat_model(self) -> BaseChatModel:  # pragma: no cover - 不调用真实模型
            raise NotImplementedError

        @property
        def name(self) -> str:
            return "dummy"

    assert isolated_registry._registry["dummy"] is DummyProvider


def test_create_dispatches_by_type_and_passes_cfg(isolated_registry):
    @isolated_registry.register("dummy")
    class DummyProvider(BaseLLMProvider):
        def __init__(self, cfg: LLMConfig) -> None:
            self.cfg = cfg

        def get_chat_model(self) -> BaseChatModel:  # pragma: no cover
            raise NotImplementedError

        @property
        def name(self) -> str:
            return "dummy"

    cfg = _make_cfg("dummy")
    provider = isolated_registry.create(cfg)

    assert isinstance(provider, DummyProvider)
    assert isinstance(provider, BaseLLMProvider)
    assert provider.cfg is cfg
    assert provider.name == "dummy"


def test_create_dispatches_among_multiple_types(isolated_registry):
    @isolated_registry.register("alpha")
    class AlphaProvider(BaseLLMProvider):
        def __init__(self, cfg: LLMConfig) -> None:
            self.cfg = cfg

        def get_chat_model(self) -> BaseChatModel:  # pragma: no cover
            raise NotImplementedError

        @property
        def name(self) -> str:
            return "alpha"

    @isolated_registry.register("beta")
    class BetaProvider(BaseLLMProvider):
        def __init__(self, cfg: LLMConfig) -> None:
            self.cfg = cfg

        def get_chat_model(self) -> BaseChatModel:  # pragma: no cover
            raise NotImplementedError

        @property
        def name(self) -> str:
            return "beta"

    assert isinstance(isolated_registry.create(_make_cfg("alpha")), AlphaProvider)
    assert isinstance(isolated_registry.create(_make_cfg("beta")), BetaProvider)


def test_create_unknown_type_raises_clear_error(isolated_registry):
    @isolated_registry.register("known")
    class KnownProvider(BaseLLMProvider):
        def __init__(self, cfg: LLMConfig) -> None:
            self.cfg = cfg

        def get_chat_model(self) -> BaseChatModel:  # pragma: no cover
            raise NotImplementedError

        @property
        def name(self) -> str:
            return "known"

    with pytest.raises(ValueError) as exc:
        isolated_registry.create(_make_cfg("does_not_exist"))

    msg = str(exc.value)
    assert "does_not_exist" in msg
    # 错误信息应列出可用类型，便于定位配置问题
    assert "known" in msg


def test_register_rejects_non_provider_subclass(isolated_registry):
    with pytest.raises(TypeError):

        @isolated_registry.register("bad")
        class NotAProvider:  # 非 BaseLLMProvider 子类
            pass


def test_register_rejects_duplicate_key(isolated_registry):
    @isolated_registry.register("dup")
    class FirstProvider(BaseLLMProvider):
        def __init__(self, cfg: LLMConfig) -> None:
            self.cfg = cfg

        def get_chat_model(self) -> BaseChatModel:  # pragma: no cover
            raise NotImplementedError

        @property
        def name(self) -> str:
            return "first"

    with pytest.raises(ValueError):

        @isolated_registry.register("dup")
        class SecondProvider(BaseLLMProvider):
            def __init__(self, cfg: LLMConfig) -> None:
                self.cfg = cfg

            def get_chat_model(self) -> BaseChatModel:  # pragma: no cover
                raise NotImplementedError

            @property
            def name(self) -> str:
                return "second"


def test_base_provider_is_abstract():
    """BaseLLMProvider 不可直接实例化（抽象方法未实现）。"""
    with pytest.raises(TypeError):
        BaseLLMProvider()  # type: ignore[abstract]
