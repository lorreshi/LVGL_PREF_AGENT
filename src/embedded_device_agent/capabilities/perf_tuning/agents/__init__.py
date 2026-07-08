"""Capability A（性能调优）的 LLM 子智能体子包。

这些子智能体（Collector / Analyzer / Instrumentor / Optimizer）由 Orchestrator
以 subagents-as-tools 模式装配调用（需求 13.1）；仅推理/判断步骤作为 LLM 子智能体，
所有副作用（串口、编译烧录、LLM 调用）均经注入接口完成，测试时可替换为 fake。
"""

from embedded_device_agent.capabilities.perf_tuning.agents.analyzer import Analyzer
from embedded_device_agent.capabilities.perf_tuning.agents.collector import (
    Collector,
    CollectResult,
)
from embedded_device_agent.capabilities.perf_tuning.agents.instrumentor import (
    Instrumentor,
)
from embedded_device_agent.capabilities.perf_tuning.agents.optimizer import Optimizer

__all__ = [
    "Collector",
    "CollectResult",
    "Analyzer",
    "Instrumentor",
    "Optimizer",
]
