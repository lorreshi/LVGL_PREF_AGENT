"""LLM Provider 后端共用工具。

集中放置各具体 provider 复用的小工具：从 ``LLMConfig.api_key_env`` 命名的环境
变量读取密钥。密钥只在需要构造真实 chat model 时（``get_chat_model``）才读取，
且从不写入配置文件 / state（对照 design.md 对 ``api_key_env`` 的说明）。
"""

from __future__ import annotations

import os

__all__ = ["require_api_key"]


def require_api_key(api_key_env: str) -> str:
    """从 ``api_key_env`` 命名的环境变量读取密钥，缺失时报出清晰错误。

    错误只暴露环境变量名，绝不回显密钥值。
    """
    if not api_key_env:
        raise ValueError("LLMConfig.api_key_env 不能为空：需指明存放密钥的环境变量名。")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            f"环境变量 '{api_key_env}' 未设置或为空，无法获取 LLM API 密钥。"
            f"请先在环境中导出该变量。"
        )
    return api_key
