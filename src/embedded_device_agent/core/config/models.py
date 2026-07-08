"""配置层 Pydantic v2 数据契约。

集中定义驱动整个系统的 YAML 配置契约，保证：

* 可独立构造、可断言，且缺字段/非法值在任何 agent 运行前即被 Pydantic
  校验报出（需求 12.1 / 12.2，具体报错定位留待 ``Config_Loader``）；
* 各子配置带 ``type`` 字段，供对应工厂（LLM / Device / Retriever）依名分发
  构造（需求 12.3 / 12.4 / 12.5）；
* ``AppConfig`` 同时承载调优闭环的预算与模式参数（帧预算、热点裁剪、
  token 预算、自动化模式、迭代上限、干预点）。

字段严格对照 design.md "Data Models" 章节的配置部分。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "LLMConfig",
    "DeviceConfig",
    "RetrieverConfig",
    "AppConfig",
]


class LLMConfig(BaseModel):
    """LLM_Provider 配置——供 LLMProviderFactory 依 ``type`` 分发构造。

    ``api_key_env`` 命名的是存放密钥的环境变量名，而非密钥本身
    （密钥不入配置文件 / state）。``base_url`` 供 openai_compatible 等
    自定义端点使用。
    """

    type: str  # anthropic / openai / openai_compatible
    model: str
    api_key_env: str
    base_url: str | None = None
    temperature: float = 0.0


class DeviceConfig(BaseModel):
    """DeviceIO 后端配置——供 DeviceIOFactory 依 ``type`` 分发构造。

    ``open_serial`` 前会校验 ``baud <= max_safe_baud``（需求 2.3）；
    ``build_cmd`` / ``flash_cmd`` 供可脚本化的构建/烧录后端使用，
    半自动/人在环场景可缺省。
    """

    type: str  # serial / half_auto / fake
    port: str | None = None
    baud: int = 921600
    max_safe_baud: int = 921600  # 需求 2.3
    build_cmd: str | None = None
    flash_cmd: str | None = None


class RetrieverConfig(BaseModel):
    """Retriever 后端配置——供 RetrieverFactory 依 ``type`` 分发构造。

    默认 ``local_store``（基于 LangGraph Store 的语义召回）；可经 YAML
    切换为 ``external_rag`` 以 tool 形式调用外部 RAG（需求 9.4 / 12.5）。
    ``top_k`` 为默认召回条数；``endpoint`` / ``collection`` 供外部 RAG
    后端使用，本地后端可缺省。
    """

    type: str = "local_store"  # local_store / external_rag
    top_k: int = 5
    endpoint: str | None = None
    collection: str | None = None


class AppConfig(BaseModel):
    """顶层应用配置：底座三件套 + 调优闭环预算与模式参数。

    由 ``Config_Loader`` 从 YAML 装配并校验后一次性构造，注入
    ``Core_Services``（需求 18.4）。
    """

    llm: LLMConfig
    device: DeviceConfig
    retriever: RetrieverConfig
    frame_budget_us: int  # 目标帧率对应预算，需求 5.1
    hotspot_top_n: int = 20  # 热点裁剪上限（上下文预算机制 2）
    report_token_budget: int = 4000  # FrameReport 序列化上限，超限二次聚合
    mode: Literal["full_auto", "half_auto"]
    max_iterations: int  # 需求 11.1
    intervention_points: list[str] = Field(default_factory=list)
