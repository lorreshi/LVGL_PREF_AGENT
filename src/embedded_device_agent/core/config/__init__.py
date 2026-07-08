"""配置层：Pydantic 配置模型与 Config_Loader。

配置模型（``models``）定义 YAML → Pydantic 的运行时契约，供各工厂
（LLM / Device / Retriever）依 ``type`` 字段分发构造，并集中承载调优
闭环的预算与模式参数。见 design.md "Data Models" 配置部分。
"""

from embedded_device_agent.core.config.models import (
    AppConfig,
    DeviceConfig,
    LLMConfig,
    RetrieverConfig,
)

__all__ = [
    "AppConfig",
    "DeviceConfig",
    "LLMConfig",
    "RetrieverConfig",
]
