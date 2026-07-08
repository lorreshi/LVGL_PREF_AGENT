"""录制的测试 fixtures（采集日志、构建/烧录结果回放、预置 LLM 决策等）。

集中暴露 fixtures 目录路径与常用录制文件路径，供 Test_Harness（``FakeDeviceIO`` /
``FakeLLMProvider``）与各能力的离线可重复测试引用（需求 20.1 / 20.2 / 20.4）。
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "FIXTURES_DIR",
    "RECORDED_CAPTURE_LOG",
    "LVGL_PROFILER_SAMPLE_LOG",
]

#: fixtures 根目录（本文件所在目录）。
FIXTURES_DIR: Path = Path(__file__).resolve().parent

#: 录制的采集日志，供 ``FakeDeviceIO.capture`` 确定性回放。
RECORDED_CAPTURE_LOG: Path = FIXTURES_DIR / "recorded_capture.log"

#: 含受损行的 LVGL profiler 样例日志，供确定性核心工具（Trace_Filter 等）独立执行。
LVGL_PROFILER_SAMPLE_LOG: Path = FIXTURES_DIR / "lvgl_profiler_sample.log"
