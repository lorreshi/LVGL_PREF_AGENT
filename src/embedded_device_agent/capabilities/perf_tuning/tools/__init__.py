"""Capability A 的确定性核心工具（可离线单测、属性测试主战场）。

这些工具均为纯函数式：对相同输入恒产生相同输出，除产出落盘工件外不产生
其他外部副作用（Property 8：确定性核心无副作用）。
"""

from embedded_device_agent.capabilities.perf_tuning.tools.trace_filter import (
    trace_filter,
)

__all__ = ["trace_filter"]
