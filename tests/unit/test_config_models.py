"""任务 2.1：配置层 Pydantic 模型的基本单元测试。

覆盖：模型构造、默认值、Literal 校验、必填字段缺失报错。
不触达真实 LLM / 串口 / 外部 RAG（纯数据契约校验）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from embedded_device_agent.core.config.models import (
    AppConfig,
    DeviceConfig,
    LLMConfig,
    RetrieverConfig,
)


# ---- LLMConfig ----
def test_llm_config_defaults():
    cfg = LLMConfig(type="anthropic", model="claude-3-5-sonnet", api_key_env="ANTHROPIC_API_KEY")
    assert cfg.base_url is None
    assert cfg.temperature == 0.0


def test_llm_config_missing_required_field_raises():
    with pytest.raises(ValidationError):
        LLMConfig(type="openai", model="gpt-4o")  # 缺 api_key_env


def test_llm_config_roundtrip():
    cfg = LLMConfig(
        type="openai_compatible",
        model="qwen",
        api_key_env="QWEN_KEY",
        base_url="http://localhost:8000/v1",
        temperature=0.2,
    )
    assert LLMConfig.model_validate_json(cfg.model_dump_json()) == cfg


# ---- DeviceConfig ----
def test_device_config_defaults():
    cfg = DeviceConfig(type="serial")
    assert cfg.port is None
    assert cfg.baud == 921600
    assert cfg.max_safe_baud == 921600
    assert cfg.build_cmd is None
    assert cfg.flash_cmd is None


def test_device_config_missing_type_raises():
    with pytest.raises(ValidationError):
        DeviceConfig()  # 缺 type


# ---- RetrieverConfig ----
def test_retriever_config_defaults():
    cfg = RetrieverConfig()
    assert cfg.type == "local_store"
    assert cfg.top_k == 5
    assert cfg.endpoint is None
    assert cfg.collection is None


def test_retriever_config_external_rag():
    cfg = RetrieverConfig(type="external_rag", endpoint="http://rag", collection="lvgl", top_k=10)
    assert cfg.type == "external_rag"
    assert cfg.top_k == 10


# ---- AppConfig ----
def _make_app_config(**overrides):
    base = dict(
        llm=LLMConfig(type="anthropic", model="claude", api_key_env="ANTHROPIC_API_KEY"),
        device=DeviceConfig(type="serial", port="/dev/ttyUSB0"),
        retriever=RetrieverConfig(),
        frame_budget_us=16667,
        mode="full_auto",
        max_iterations=5,
    )
    base.update(overrides)
    return AppConfig(**base)


def test_app_config_defaults():
    cfg = _make_app_config()
    assert cfg.hotspot_top_n == 20
    assert cfg.report_token_budget == 4000
    assert cfg.intervention_points == []


def test_app_config_intervention_points_independent_instances():
    a = _make_app_config()
    b = _make_app_config()
    a.intervention_points.append("collect")
    assert b.intervention_points == []  # default_factory 避免共享可变默认值


def test_app_config_invalid_mode_raises():
    with pytest.raises(ValidationError):
        _make_app_config(mode="semi_auto")  # 非 Literal 值


def test_app_config_missing_required_field_raises():
    with pytest.raises(ValidationError):
        AppConfig(
            llm=LLMConfig(type="anthropic", model="claude", api_key_env="K"),
            device=DeviceConfig(type="serial"),
            retriever=RetrieverConfig(),
            mode="half_auto",
            max_iterations=3,
        )  # 缺 frame_budget_us


def test_app_config_roundtrip():
    cfg = _make_app_config(mode="half_auto", intervention_points=["collect", "optimize"])
    assert AppConfig.model_validate_json(cfg.model_dump_json()) == cfg
