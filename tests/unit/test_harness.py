"""任务 16.1：Test_Harness 的 FakeDeviceIO / FakeLLMProvider 单元测试（需求 20.1/20.2/20.4）。

验证两点：
* 两个 fake 均**符合各自接口契约**（``DeviceIO`` / ``BaseLLMProvider``）；
* 回放**确定性**——同样入参多次调用产出一致，且不触达真实硬件 / LLM。
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from embedded_device_agent.core.device.base import DeviceIO
from embedded_device_agent.core.device.models import (
    BuildResult,
    FlashResult,
    InputEvent,
)
from embedded_device_agent.core.llm.base import BaseLLMProvider
from embedded_device_agent.core.models import RawTraceArtifact
from tests.fixtures import RECORDED_CAPTURE_LOG
from tests.harness import FakeDeviceIO, FakeLLMProvider


# ---------------------------------------------------------------------------
# 需求 20.1：FakeDeviceIO 实现 DeviceIO，回放录制日志与构建/烧录结果
# ---------------------------------------------------------------------------
def test_fake_device_io_conforms_to_interface() -> None:
    dev = FakeDeviceIO()
    assert isinstance(dev, DeviceIO)


def test_capture_returns_artifact_pointing_at_recorded_fixture() -> None:
    dev = FakeDeviceIO()
    artifact = dev.capture(duration_s=1.5)

    assert isinstance(artifact, RawTraceArtifact)
    assert artifact.path == RECORDED_CAPTURE_LOG
    assert artifact.path.exists()  # 录制 fixture 真实存在
    assert artifact.duration_s == 1.5
    assert artifact.baud == 115200


def test_capture_is_deterministic_across_instances() -> None:
    a = FakeDeviceIO().capture(duration_s=2.0)
    b = FakeDeviceIO().capture(duration_s=2.0)
    # 相同入参 → 完全一致的工件（run_id / 时间戳 / 路径均确定性）。
    assert a.model_dump() == b.model_dump()


def test_capture_run_id_increments_within_instance() -> None:
    dev = FakeDeviceIO()
    first = dev.capture(duration_s=1.0)
    second = dev.capture(duration_s=1.0)
    assert first.run_id != second.run_id
    assert first.run_id == "fake-run-0000"
    assert second.run_id == "fake-run-0001"


def test_build_and_flash_replay_preset_results_in_order() -> None:
    dev = FakeDeviceIO(
        build_results=[
            BuildResult(success=False, error="boom"),
            BuildResult(success=True, output="fixed"),
        ],
        flash_results=[FlashResult(success=True, output="flashed")],
    )

    first = dev.build()
    second = dev.build()
    third = dev.build()  # 耗尽后重复最后一项

    assert first.success is False and first.error == "boom"
    assert second.success is True and second.output == "fixed"
    assert third.model_dump() == second.model_dump()

    flash = dev.flash()
    assert isinstance(flash, FlashResult)
    assert flash.success is True and flash.output == "flashed"


def test_build_flash_defaults_are_successful() -> None:
    dev = FakeDeviceIO()
    assert dev.build().success is True
    assert dev.flash().success is True


def test_send_cmd_and_inject_input_are_recorded_without_hardware() -> None:
    dev = FakeDeviceIO(cmd_responses=["pong", "ready"])

    assert dev.send_cmd("ping") == "pong"
    assert dev.send_cmd("status") == "ready"
    assert dev.send_cmd("again") == "ready"  # 耗尽后重复最后一项

    dev.open_serial("/dev/fake", 115200)
    dev.inject_input(InputEvent(kind="touch", params={"x": 1, "y": 2}))

    assert dev.opened == ("/dev/fake", 115200)
    assert dev.injected_inputs == [InputEvent(kind="touch", params={"x": 1, "y": 2})]
    assert ("send_cmd", "ping") in dev.calls
    assert ("capture", None) not in dev.calls  # 未调用 capture


# ---------------------------------------------------------------------------
# 需求 20.2：FakeLLMProvider 实现 BaseLLMProvider，回放预置决策
# ---------------------------------------------------------------------------
def test_fake_llm_provider_conforms_to_interface() -> None:
    provider = FakeLLMProvider(responses=["decision"])
    assert isinstance(provider, BaseLLMProvider)
    assert provider.name == "fake"
    assert FakeLLMProvider(responses=["x"], name="offline").name == "offline"


def test_fake_llm_provider_returns_chat_model_replaying_presets() -> None:
    provider = FakeLLMProvider(responses=["首个决策", "第二个决策"])
    model = provider.get_chat_model()
    assert isinstance(model, BaseChatModel)

    first = model.invoke([HumanMessage(content="hi")])
    second = model.invoke([HumanMessage(content="again")])
    assert first.content == "首个决策"
    assert second.content == "第二个决策"


def test_fake_llm_provider_is_deterministic_across_models() -> None:
    provider = FakeLLMProvider(responses=["a", "b"])
    out1 = provider.get_chat_model().invoke([HumanMessage(content="q")]).content
    out2 = provider.get_chat_model().invoke([HumanMessage(content="q")]).content
    assert out1 == out2 == "a"


def test_fake_llm_provider_requires_responses() -> None:
    import pytest

    with pytest.raises(ValueError):
        FakeLLMProvider(responses=[])
