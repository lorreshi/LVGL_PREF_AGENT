"""Capability A 的确定性核心工具（可离线单测、属性测试主战场）。

这些工具均为纯函数式：对相同输入恒产生相同输出，除产出落盘工件外不产生
其他外部副作用（Property 8：确定性核心无副作用）。
"""

from embedded_device_agent.capabilities.perf_tuning.tools.systrace_parser import (
    iter_systrace_events,
    parse_systrace,
    serialize_parse_result,
)
from embedded_device_agent.capabilities.perf_tuning.tools.trace_filter import (
    trace_filter,
)

__all__ = [
    "trace_filter",
    "parse_systrace",
    "iter_systrace_events",
    "serialize_parse_result",
]
