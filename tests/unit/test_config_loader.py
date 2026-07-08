"""任务 2.2：Config_Loader 基础单元测试。

覆盖：
* 合法 YAML 装配成功（含默认值填充）；
* 缺失 / 非法字段能报出**具体字段与原因**（需求 12.2）；
* 文件不存在、YAML 语法错误、顶层结构非映射等边界；
* 随附的 config/config.example.yaml 示例可被成功加载。

不触达真实 LLM / 串口 / 外部 RAG（纯配置装配与校验）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from embedded_device_agent.core.config.loader import (
    ConfigError,
    ConfigLoader,
    load_config,
)
from embedded_device_agent.core.config.models import AppConfig

# 仓库根：tests/unit/test_config_loader.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "config.example.yaml"

VALID_YAML = """
llm:
  type: anthropic
  model: claude-3-5-sonnet
  api_key_env: ANTHROPIC_API_KEY
device:
  type: serial
  port: /dev/ttyUSB0
  baud: 921600
  max_safe_baud: 921600
retriever:
  type: local_store
  top_k: 5
frame_budget_us: 16667
mode: full_auto
max_iterations: 5
"""


# ---- 合法装配 ----
def test_from_string_valid_yaml_builds_appconfig():
    cfg = ConfigLoader.from_string(VALID_YAML)
    assert isinstance(cfg, AppConfig)
    assert cfg.llm.type == "anthropic"
    assert cfg.device.port == "/dev/ttyUSB0"
    assert cfg.frame_budget_us == 16667
    assert cfg.mode == "full_auto"
    assert cfg.max_iterations == 5
    # 默认值被正确填充
    assert cfg.hotspot_top_n == 20
    assert cfg.report_token_budget == 4000
    assert cfg.intervention_points == []


def test_from_file_reads_and_validates(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML, encoding="utf-8")
    cfg = load_config(p)
    assert isinstance(cfg, AppConfig)
    assert cfg.retriever.top_k == 5


def test_example_config_loads_successfully():
    """随任务交付的示例配置必须能被成功加载与校验。"""
    assert EXAMPLE_CONFIG.is_file(), f"示例配置缺失：{EXAMPLE_CONFIG}"
    cfg = ConfigLoader.from_file(EXAMPLE_CONFIG)
    assert isinstance(cfg, AppConfig)
    assert cfg.mode in ("full_auto", "half_auto")
    assert cfg.max_iterations >= 1


# ---- 缺失字段：报出具体字段与原因 ----
def test_missing_required_field_reports_field_and_reason():
    bad = """
llm:
  type: anthropic
  model: claude
  api_key_env: K
device:
  type: serial
retriever:
  type: local_store
mode: full_auto
max_iterations: 5
"""  # 缺 frame_budget_us
    with pytest.raises(ConfigError) as ei:
        ConfigLoader.from_string(bad)
    msg = str(ei.value)
    assert "frame_budget_us" in msg  # 精确到具体字段
    assert "校验失败" in msg


def test_missing_nested_field_reports_dotted_path():
    bad = """
llm:
  type: anthropic
  model: claude
device:
  type: serial
retriever:
  type: local_store
frame_budget_us: 16667
mode: full_auto
max_iterations: 5
"""  # llm 缺 api_key_env
    with pytest.raises(ConfigError) as ei:
        ConfigLoader.from_string(bad)
    assert "llm.api_key_env" in str(ei.value)  # 嵌套字段用点号路径定位


# ---- 非法值：报出具体字段与原因 ----
def test_invalid_mode_reports_field():
    bad = VALID_YAML.replace("mode: full_auto", "mode: semi_auto")
    with pytest.raises(ConfigError) as ei:
        ConfigLoader.from_string(bad)
    assert "mode" in str(ei.value)


def test_invalid_type_for_int_reports_field():
    bad = VALID_YAML.replace("max_iterations: 5", "max_iterations: not_a_number")
    with pytest.raises(ConfigError) as ei:
        ConfigLoader.from_string(bad)
    assert "max_iterations" in str(ei.value)


# ---- 边界：文件/语法/结构 ----
def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError) as ei:
        ConfigLoader.from_file(tmp_path / "nope.yaml")
    assert "不存在" in str(ei.value)


def test_invalid_yaml_syntax_raises_config_error():
    with pytest.raises(ConfigError) as ei:
        ConfigLoader.from_string("llm: [unclosed\n  : :")
    assert "YAML" in str(ei.value)


def test_empty_yaml_raises_config_error():
    with pytest.raises(ConfigError):
        ConfigLoader.from_string("")


def test_non_mapping_top_level_raises_config_error():
    with pytest.raises(ConfigError) as ei:
        ConfigLoader.from_string("- just\n- a\n- list")
    assert "映射" in str(ei.value)
