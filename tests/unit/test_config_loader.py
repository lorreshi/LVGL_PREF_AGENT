"""任务 2.3：Config_Loader 单元测试 + Property 7（配置先校验后运行）。

覆盖两类内容：

* **单元测试**：合法 YAML 装配、以及缺失 / 非法字段时 :class:`ConfigError`
  的具体报错定位（字段路径 + 原因），对应需求 12.1 / 12.2。
* **属性测试（Property 7）**：任何 agent / 工厂在 ``Config_Loader`` 校验通过
  前不得被构造或执行。用一个带副作用计数的 spy 工厂验证：校验失败时其调用
  次数恒为 0；仅当校验通过后才被调用一次，且收到的是已校验的 ``AppConfig``。

所有断言均基于真实的 :class:`ConfigLoader` 行为——校验失败时它在返回
``AppConfig`` **之前**抛出 ``ConfigError``，因此任何遵循「先校验后构造」
装配模式的下游代码都无法在校验未过时产生副作用。
"""

from __future__ import annotations

from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from embedded_device_agent.core.config.loader import (
    ConfigError,
    ConfigLoader,
    load_config,
)
from embedded_device_agent.core.config.models import AppConfig

# ---------------------------------------------------------------------------
# 公共 fixture：一份最小合法配置（映射形式）
# ---------------------------------------------------------------------------


def _valid_mapping() -> dict:
    return {
        "llm": {
            "type": "anthropic",
            "model": "claude-3-5-sonnet-latest",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        "device": {"type": "fake"},
        "retriever": {"type": "local_store", "top_k": 5},
        "frame_budget_us": 16667,
        "mode": "full_auto",
        "max_iterations": 5,
    }


_VALID_YAML = """
llm:
  type: anthropic
  model: claude-3-5-sonnet-latest
  api_key_env: ANTHROPIC_API_KEY
device:
  type: fake
retriever:
  type: local_store
  top_k: 5
frame_budget_us: 16667
mode: full_auto
max_iterations: 5
"""


# ---------------------------------------------------------------------------
# 单元测试：合法装配
# ---------------------------------------------------------------------------


def test_from_mapping_valid_builds_appconfig():
    cfg = ConfigLoader.from_mapping(_valid_mapping())
    assert isinstance(cfg, AppConfig)
    assert cfg.llm.type == "anthropic"
    assert cfg.device.type == "fake"
    assert cfg.mode == "full_auto"
    assert cfg.max_iterations == 5


def test_from_string_valid_yaml():
    cfg = ConfigLoader.from_string(_VALID_YAML)
    assert isinstance(cfg, AppConfig)
    assert cfg.frame_budget_us == 16667
    # 默认值随契约填充
    assert cfg.hotspot_top_n == 20
    assert cfg.report_token_budget == 4000


def test_from_file_valid_yaml(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_VALID_YAML, encoding="utf-8")
    cfg = ConfigLoader.from_file(cfg_path)
    assert isinstance(cfg, AppConfig)
    # 便捷函数等价
    assert load_config(cfg_path).model_dump() == cfg.model_dump()


# ---------------------------------------------------------------------------
# 单元测试：缺失 / 非法字段的具体报错定位（需求 12.2）
# ---------------------------------------------------------------------------


def test_missing_top_level_section_reports_field():
    data = _valid_mapping()
    del data["llm"]
    with pytest.raises(ConfigError) as exc:
        ConfigLoader.from_mapping(data)
    msg = str(exc.value)
    assert "校验失败" in msg
    assert "llm" in msg  # 字段路径明确指向缺失项


def test_missing_nested_required_field_reports_dotted_path():
    data = _valid_mapping()
    del data["llm"]["api_key_env"]
    with pytest.raises(ConfigError) as exc:
        ConfigLoader.from_mapping(data)
    msg = str(exc.value)
    # 嵌套字段路径以点号拼接，精确到 llm.api_key_env
    assert "llm.api_key_env" in msg


def test_invalid_mode_literal_reports_field():
    data = _valid_mapping()
    data["mode"] = "turbo"  # 不在 Literal[full_auto, half_auto]
    with pytest.raises(ConfigError) as exc:
        ConfigLoader.from_mapping(data)
    assert "mode" in str(exc.value)


def test_invalid_type_for_max_iterations_reports_field():
    data = _valid_mapping()
    data["max_iterations"] = ["not", "an", "int"]  # 不可强制转换
    with pytest.raises(ConfigError) as exc:
        ConfigLoader.from_mapping(data)
    assert "max_iterations" in str(exc.value)


def test_multiple_errors_all_reported():
    data = _valid_mapping()
    del data["llm"]
    del data["device"]
    data["mode"] = "bogus"
    with pytest.raises(ConfigError) as exc:
        ConfigLoader.from_mapping(data)
    msg = str(exc.value)
    # 逐条列出，三处问题均出现在报错中
    assert "llm" in msg
    assert "device" in msg
    assert "mode" in msg


def test_missing_file_reports_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError) as exc:
        ConfigLoader.from_file(missing)
    assert "不存在" in str(exc.value)


def test_yaml_syntax_error_reports_source():
    bad_yaml = "llm: [unclosed\n  : :"
    with pytest.raises(ConfigError) as exc:
        ConfigLoader.from_string(bad_yaml)
    assert "YAML 语法错误" in str(exc.value)


def test_empty_config_rejected():
    with pytest.raises(ConfigError) as exc:
        ConfigLoader.from_string("")
    assert "配置为空" in str(exc.value)


def test_non_mapping_top_level_rejected():
    with pytest.raises(ConfigError) as exc:
        ConfigLoader.from_string("- a\n- b\n")  # 顶层为列表
    assert "顶层结构非法" in str(exc.value)


# ---------------------------------------------------------------------------
# Property 7: 配置先校验后运行
# 任何 agent / 工厂在 Config_Loader 校验通过前不得被构造或执行。
# **Validates: Requirements 12.1, 12.2**
# ---------------------------------------------------------------------------


class _ConstructionSpy:
    """带副作用计数的工厂替身：每次「构造组件」即自增计数。

    用于证明校验未通过时其从未被调用（副作用未发生）。
    """

    def __init__(self) -> None:
        self.count = 0
        self.received: AppConfig | None = None

    def __call__(self, cfg: AppConfig):
        self.count += 1
        self.received = cfg
        return object()  # 代表被构造出来的组件


def _guarded_assemble(raw: dict, factory: _ConstructionSpy):
    """production 装配所遵循的「先校验后构造」模式的最小复现：

    先经 ``Config_Loader`` 校验（失败即抛 ``ConfigError`` 且**不返回**），
    仅当校验通过后才调用工厂构造组件。
    """
    cfg = ConfigLoader.from_mapping(raw)  # 校验失败在此抛出，不会往下走
    return factory(cfg)


# 生成器：合法配置映射
_llm_types = st.sampled_from(["anthropic", "openai", "openai_compatible"])
_device_types = st.sampled_from(["serial", "half_auto", "fake"])
_modes = st.sampled_from(["full_auto", "half_auto"])
_short_text = st.text(min_size=1, max_size=24)


@st.composite
def _valid_config_dicts(draw) -> dict:
    return {
        "llm": {
            "type": draw(_llm_types),
            "model": draw(_short_text),
            "api_key_env": draw(_short_text),
        },
        "device": {"type": draw(_device_types)},
        "retriever": {"type": "local_store"},
        "frame_budget_us": draw(st.integers(min_value=1, max_value=10**9)),
        "hotspot_top_n": draw(st.integers(min_value=1, max_value=1000)),
        "report_token_budget": draw(st.integers(min_value=1, max_value=10**6)),
        "mode": draw(_modes),
        "max_iterations": draw(st.integers(min_value=1, max_value=100)),
    }


# 无默认值的必填顶层字段——去掉任一必然导致校验失败
_REQUIRED_TOP = ["llm", "device", "retriever", "frame_budget_us", "mode", "max_iterations"]


@st.composite
def _invalid_config_dicts(draw) -> dict:
    cfg = draw(_valid_config_dicts())
    victim = draw(st.sampled_from(_REQUIRED_TOP))
    cfg.pop(victim, None)
    return cfg


@settings(max_examples=100)
@given(_invalid_config_dicts())
def test_property7_no_construction_when_validation_fails(raw: dict):
    """校验失败 ⇒ 抛 ConfigError 且工厂调用计数恒为 0（无组件被构造）。"""
    spy = _ConstructionSpy()
    with pytest.raises(ConfigError):
        _guarded_assemble(raw, spy)
    assert spy.count == 0
    assert spy.received is None


@settings(max_examples=100)
@given(_valid_config_dicts())
def test_property7_construction_only_after_validation_passes(raw: dict):
    """校验通过 ⇒ 工厂恰被调用一次，且收到已校验的 AppConfig。"""
    spy = _ConstructionSpy()
    result = _guarded_assemble(raw, spy)
    assert result is not None
    assert spy.count == 1
    assert isinstance(spy.received, AppConfig)
