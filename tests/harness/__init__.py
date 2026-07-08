"""确定性测试 harness：FakeDeviceIO / FakeLLMProvider / 录制 fixtures（需求 20）。

对照 design.md "Test_Harness"：以可回放的 ``FakeDeviceIO`` + ``FakeLLMProvider`` +
``tests/fixtures/`` 下的录制 fixtures 组成确定性试验台，使任意能力都能脱离真实
硬件与真实 LLM 离线、可重复地自测（InMemoryStore 由后续任务补充）。
"""

from __future__ import annotations

from tests.harness.fakes import FakeDeviceIO, FakeLLMProvider

__all__ = ["FakeDeviceIO", "FakeLLMProvider"]
