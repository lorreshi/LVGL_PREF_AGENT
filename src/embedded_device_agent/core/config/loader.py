"""Config_Loader：YAML 加载 → Pydantic 校验（需求 12.1 / 12.2 / 11.1）。

职责（对照 design.md "Config_Loader" 章节）：

* 启动时从 YAML 读取配置并交由 :class:`AppConfig` 做 Pydantic 校验；
* 在**任何 agent 运行前**，一旦有必填字段缺失或值非法，即抛出
  :class:`ConfigError`，其消息**逐条列出具体字段路径与原因**，实现清晰的
  错误定位（需求 12.2）；
* 自动化模式（``mode``）与迭代上限（``max_iterations``）随 ``AppConfig``
  一并校验读取（需求 11.1）。

设计取舍：文件不存在、YAML 语法错误、顶层结构非映射、Pydantic 校验失败
四类问题都归一化为 :class:`ConfigError`，调用方（Core_Services 装配）只需
捕获单一异常类型即可在启动阶段拦截全部配置问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .models import AppConfig

__all__ = ["ConfigError", "ConfigLoader", "load_config"]


class ConfigError(Exception):
    """配置加载 / 校验失败。

    消息面向使用者，逐条给出「字段路径: 原因」，以便在任何 agent 运行前
    精确定位问题（需求 12.2）。
    """


def _format_validation_error(exc: ValidationError, *, source: str) -> str:
    """把 Pydantic ``ValidationError`` 展开成逐字段的可读定位信息。"""

    lines: list[str] = [
        f"配置校验失败（{source}），共 {exc.error_count()} 处问题："
    ]
    for err in exc.errors():
        # ``loc`` 形如 ('llm', 'api_key_env')；用点号拼成字段路径。
        loc = err.get("loc", ())
        field_path = ".".join(str(part) for part in loc) if loc else "<root>"
        reason = err.get("msg", "无效值")
        lines.append(f"  - 字段 '{field_path}': {reason}")
    return "\n".join(lines)


class ConfigLoader:
    """从 YAML 文件加载并校验 :class:`AppConfig` 的加载器。

    Example
    -------
    >>> cfg = ConfigLoader.from_file("config/config.yaml")  # doctest: +SKIP
    """

    @staticmethod
    def from_file(path: str | Path) -> AppConfig:
        """读取 YAML 文件、解析并校验，返回构造好的 :class:`AppConfig`。

        Raises
        ------
        ConfigError
            文件不存在、YAML 语法错误、顶层结构非映射，或任何必填字段缺失
            / 值非法时抛出，消息含具体字段与原因。
        """

        cfg_path = Path(path)
        if not cfg_path.is_file():
            raise ConfigError(f"配置文件不存在：{cfg_path}")

        try:
            raw_text = cfg_path.read_text(encoding="utf-8")
        except OSError as exc:  # 权限 / IO 错误
            raise ConfigError(f"无法读取配置文件 {cfg_path}：{exc}") from exc

        return ConfigLoader.from_string(raw_text, source=str(cfg_path))

    @staticmethod
    def from_string(text: str, *, source: str = "<string>") -> AppConfig:
        """从 YAML 文本解析并校验，返回 :class:`AppConfig`。"""

        try:
            data: Any = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML 语法错误（{source}）：{exc}") from exc

        if data is None:
            raise ConfigError(f"配置为空（{source}）：需要一个包含配置字段的映射")
        if not isinstance(data, dict):
            raise ConfigError(
                f"配置顶层结构非法（{source}）：期望映射，实际为 {type(data).__name__}"
            )

        return ConfigLoader.from_mapping(data, source=source)

    @staticmethod
    def from_mapping(data: dict[str, Any], *, source: str = "<mapping>") -> AppConfig:
        """从已解析的映射校验并构造 :class:`AppConfig`。"""

        try:
            return AppConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(exc, source=source)) from exc


def load_config(path: str | Path) -> AppConfig:
    """便捷函数：等价于 :meth:`ConfigLoader.from_file`。"""

    return ConfigLoader.from_file(path)
