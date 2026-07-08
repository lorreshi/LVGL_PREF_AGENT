"""具体 LLM Provider 后端单元测试（任务 3.2，需求 12.3）。

只用 mock，绝不发起真实网络调用：

* 验证 import 触发注册，工厂按 ``type`` 构造出正确 provider 子类；
* 验证从 ``api_key_env`` 命名的环境变量读取密钥、正确读取 model/base_url/
  temperature，并把参数正确传给底层 ChatAnthropic/ChatOpenAI（构造被 patch）；
* 验证 ``get_chat_model()`` 返回底层 chat model 对象；
* 验证缺失环境变量、openai_compatible 缺 base_url、未知 type 的报错。

真实 LLM 连通性测试留待后续（用户提供 API key 后补跑），见文件末尾的
``test_real_*`` 骨架（缺 key 时 skip，绝不 fail）。
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from embedded_device_agent.core.config.models import LLMConfig
from embedded_device_agent.core.llm import LLMProviderFactory
from embedded_device_agent.core.llm.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)

# 各 provider 模块内 ChatAnthropic/ChatOpenAI 的可 patch 路径
_ANTHROPIC_PATH = "embedded_device_agent.core.llm.providers.anthropic.ChatAnthropic"
_OPENAI_PATH = "embedded_device_agent.core.llm.providers.openai.ChatOpenAI"
_COMPAT_PATH = "embedded_device_agent.core.llm.providers.openai_compatible.ChatOpenAI"


# ---------------------------------------------------------------------------
# 工厂按 type 分发构造正确子类（import 已触发注册）
# ---------------------------------------------------------------------------
def test_factory_dispatches_anthropic() -> None:
    cfg = LLMConfig(type="anthropic", model="claude-x", api_key_env="ANTHROPIC_API_KEY")
    provider = LLMProviderFactory.create(cfg)
    assert isinstance(provider, AnthropicProvider)
    assert provider.name == "anthropic"


def test_factory_dispatches_openai() -> None:
    cfg = LLMConfig(type="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY")
    provider = LLMProviderFactory.create(cfg)
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "openai"


def test_factory_dispatches_openai_compatible() -> None:
    cfg = LLMConfig(
        type="openai_compatible",
        model="qwen",
        api_key_env="GW_API_KEY",
        base_url="http://gw.local/v1",
    )
    provider = LLMProviderFactory.create(cfg)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "openai_compatible"


def test_factory_unknown_type_raises() -> None:
    cfg = LLMConfig(type="nope", model="m", api_key_env="X")
    with pytest.raises(ValueError, match="未知的 LLM provider type"):
        LLMProviderFactory.create(cfg)


# ---------------------------------------------------------------------------
# get_chat_model：读 env 密钥 + 正确传参给底层构造（mock，不发真实调用）
# ---------------------------------------------------------------------------
def test_anthropic_get_chat_model_reads_env_and_passes_args() -> None:
    cfg = LLMConfig(
        type="anthropic",
        model="claude-3-5-sonnet",
        api_key_env="MY_ANTHROPIC_KEY",
        temperature=0.3,
    )
    provider = AnthropicProvider(cfg)
    with patch(_ANTHROPIC_PATH) as mock_cls, patch.dict(
        os.environ, {"MY_ANTHROPIC_KEY": "secret-a"}, clear=False
    ):
        model = provider.get_chat_model()
    mock_cls.assert_called_once_with(
        model="claude-3-5-sonnet", api_key="secret-a", temperature=0.3
    )
    assert model is mock_cls.return_value


def test_openai_get_chat_model_reads_env_and_passes_args() -> None:
    cfg = LLMConfig(
        type="openai",
        model="gpt-4o",
        api_key_env="MY_OPENAI_KEY",
        temperature=0.0,
    )
    provider = OpenAIProvider(cfg)
    with patch(_OPENAI_PATH) as mock_cls, patch.dict(
        os.environ, {"MY_OPENAI_KEY": "secret-o"}, clear=False
    ):
        model = provider.get_chat_model()
    mock_cls.assert_called_once_with(
        model="gpt-4o", api_key="secret-o", temperature=0.0
    )
    assert model is mock_cls.return_value


def test_openai_passes_base_url_when_configured() -> None:
    cfg = LLMConfig(
        type="openai",
        model="gpt-4o",
        api_key_env="MY_OPENAI_KEY",
        base_url="http://proxy.local/v1",
        temperature=0.1,
    )
    provider = OpenAIProvider(cfg)
    with patch(_OPENAI_PATH) as mock_cls, patch.dict(
        os.environ, {"MY_OPENAI_KEY": "secret-o"}, clear=False
    ):
        provider.get_chat_model()
    mock_cls.assert_called_once_with(
        model="gpt-4o",
        api_key="secret-o",
        temperature=0.1,
        base_url="http://proxy.local/v1",
    )


def test_openai_compatible_get_chat_model_passes_base_url() -> None:
    cfg = LLMConfig(
        type="openai_compatible",
        model="qwen-max",
        api_key_env="GW_KEY",
        base_url="http://gw.local/v1",
        temperature=0.2,
    )
    provider = OpenAICompatibleProvider(cfg)
    with patch(_COMPAT_PATH) as mock_cls, patch.dict(
        os.environ, {"GW_KEY": "secret-g"}, clear=False
    ):
        model = provider.get_chat_model()
    mock_cls.assert_called_once_with(
        model="qwen-max",
        api_key="secret-g",
        base_url="http://gw.local/v1",
        temperature=0.2,
    )
    assert model is mock_cls.return_value


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------
def test_missing_env_var_raises_without_leaking_value() -> None:
    cfg = LLMConfig(type="openai", model="gpt-4o", api_key_env="DEFINITELY_MISSING_KEY")
    provider = OpenAIProvider(cfg)
    with patch(_OPENAI_PATH), patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DEFINITELY_MISSING_KEY", None)
        with pytest.raises(ValueError, match="DEFINITELY_MISSING_KEY"):
            provider.get_chat_model()


def test_openai_compatible_requires_base_url() -> None:
    cfg = LLMConfig(
        type="openai_compatible", model="qwen", api_key_env="GW_KEY"
    )  # 无 base_url
    provider = OpenAICompatibleProvider(cfg)
    with patch(_COMPAT_PATH), patch.dict(
        os.environ, {"GW_KEY": "secret-g"}, clear=False
    ):
        with pytest.raises(ValueError, match="base_url"):
            provider.get_chat_model()


# ---------------------------------------------------------------------------
# 真实 LLM 连通性测试（延后）：默认恒 skip，绝不 fail，也绝不发起真实调用。
#
# 本任务不做任何真实调用，故这些测试需**显式 opt-in** 才运行：同时满足
#   1) 设置 RUN_REAL_LLM_TESTS=1（显式开关，防止本地/CI 因环境里恰好存在 key
#      而误发真实请求）；且
#   2) 对应的 API key 环境变量已设置。
# 用户后续补跑真实连通性时，导出 key 并设置 RUN_REAL_LLM_TESTS=1 即可。
# ---------------------------------------------------------------------------
_REAL_LLM_OPT_IN = os.environ.get("RUN_REAL_LLM_TESTS") == "1"


@pytest.mark.skipif(
    not (_REAL_LLM_OPT_IN and os.environ.get("ANTHROPIC_API_KEY")),
    reason="真实 Anthropic 连通性测试延后：需 RUN_REAL_LLM_TESTS=1 且 ANTHROPIC_API_KEY",
)
def test_real_anthropic_invoke() -> None:  # pragma: no cover - 需真实 key
    cfg = LLMConfig(
        type="anthropic",
        model="claude-3-5-sonnet-latest",
        api_key_env="ANTHROPIC_API_KEY",
    )
    model = LLMProviderFactory.create(cfg).get_chat_model()
    resp = model.invoke("ping")
    assert resp is not None


@pytest.mark.skipif(
    not (_REAL_LLM_OPT_IN and os.environ.get("OPENAI_API_KEY")),
    reason="真实 OpenAI 连通性测试延后：需 RUN_REAL_LLM_TESTS=1 且 OPENAI_API_KEY",
)
def test_real_openai_invoke() -> None:  # pragma: no cover - 需真实 key
    cfg = LLMConfig(type="openai", model="gpt-4o-mini", api_key_env="OPENAI_API_KEY")
    model = LLMProviderFactory.create(cfg).get_chat_model()
    resp = model.invoke("ping")
    assert resp is not None
