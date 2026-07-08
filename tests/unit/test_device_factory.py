"""任务 4.1：DeviceIO 底座（DeviceIO + HumanInLoopMixin + DeviceIOFactory）单元测试。

验证工厂机制、抽象契约与人在环降级行为，不触达真实串口：
* 装饰器注册使子类进入注册表；
* ``create(cfg)`` 依 ``cfg.type`` 分发构造正确子类，并把 cfg 透传给构造函数；
* 未知 ``type`` 报出清晰错误（列出可用类型）；
* 非 DeviceIO 子类注册被拒绝、重复注册同一 key 被拒绝；
* ``DeviceIO`` 抽象不可直接实例化；
* ``HumanInLoopMixin`` 在 build/flash/inject 时返回 ``InterventionRequest`` 而非抛错；
* 辅助数据类型（BuildResult/FlashResult/InputEvent/InterventionRequest）可构造与 JSON round-trip。

用一个 DummyDeviceIO 注册来测工厂机制（无需真实串口）。
_Requirements: 14.1, 14.2, 14.3, 12.4_
"""

from __future__ import annotations

import pytest

from embedded_device_agent.core.config.models import DeviceConfig
from embedded_device_agent.core.device import (
    BuildResult,
    DeviceIO,
    DeviceIOFactory,
    FlashResult,
    HumanInLoopMixin,
    InputEvent,
    InterventionRequest,
)
from embedded_device_agent.core.models import RawTraceArtifact


@pytest.fixture
def isolated_registry():
    """隔离工厂注册表：每个用例前后清空并还原，避免用例间串扰。"""
    saved = dict(DeviceIOFactory._registry)
    DeviceIOFactory._registry.clear()
    try:
        yield DeviceIOFactory
    finally:
        DeviceIOFactory._registry.clear()
        DeviceIOFactory._registry.update(saved)


def _make_cfg(type_: str) -> DeviceConfig:
    return DeviceConfig(type=type_, port="/dev/ttyUSB0", baud=115200)


class _DummyDeviceIO(DeviceIO):
    """最小可实例化的 DeviceIO，用于测工厂机制（不触达真实硬件）。"""

    def __init__(self, cfg: DeviceConfig) -> None:
        self.cfg = cfg

    def open_serial(self, port: str, baud: int) -> None:  # pragma: no cover - 空实现
        return None

    def capture(self, duration_s: float) -> RawTraceArtifact:  # pragma: no cover
        raise NotImplementedError

    def build(self) -> BuildResult:  # pragma: no cover
        return BuildResult(success=True)

    def flash(self) -> FlashResult:  # pragma: no cover
        return FlashResult(success=True)

    def inject_input(self, event: InputEvent) -> None:  # pragma: no cover
        return None

    def send_cmd(self, cmd: str) -> str:  # pragma: no cover
        return "ok"


# ---------------------------------------------------------------------------
# 工厂：注册与按 type 分发
# ---------------------------------------------------------------------------
def test_register_decorator_adds_to_registry(isolated_registry):
    @isolated_registry.register("dummy")
    class Dummy(_DummyDeviceIO):
        pass

    assert isolated_registry._registry["dummy"] is Dummy


def test_create_dispatches_by_type_and_passes_cfg(isolated_registry):
    @isolated_registry.register("dummy")
    class Dummy(_DummyDeviceIO):
        pass

    cfg = _make_cfg("dummy")
    device = isolated_registry.create(cfg)

    assert isinstance(device, Dummy)
    assert isinstance(device, DeviceIO)
    assert device.cfg is cfg


def test_create_dispatches_among_multiple_types(isolated_registry):
    @isolated_registry.register("alpha")
    class Alpha(_DummyDeviceIO):
        pass

    @isolated_registry.register("beta")
    class Beta(_DummyDeviceIO):
        pass

    assert isinstance(isolated_registry.create(_make_cfg("alpha")), Alpha)
    assert isinstance(isolated_registry.create(_make_cfg("beta")), Beta)


def test_create_unknown_type_raises_clear_error(isolated_registry):
    @isolated_registry.register("known")
    class Known(_DummyDeviceIO):
        pass

    with pytest.raises(ValueError) as exc:
        isolated_registry.create(_make_cfg("does_not_exist"))

    msg = str(exc.value)
    assert "does_not_exist" in msg
    # 错误信息应列出可用类型，便于定位配置问题
    assert "known" in msg


def test_register_rejects_non_deviceio_subclass(isolated_registry):
    with pytest.raises(TypeError):

        @isolated_registry.register("bad")
        class NotADevice:  # 非 DeviceIO 子类
            pass


def test_register_rejects_duplicate_key(isolated_registry):
    @isolated_registry.register("dup")
    class First(_DummyDeviceIO):
        pass

    with pytest.raises(ValueError):

        @isolated_registry.register("dup")
        class Second(_DummyDeviceIO):
            pass


def test_deviceio_is_abstract():
    """DeviceIO 不可直接实例化（抽象方法未实现）。"""
    with pytest.raises(TypeError):
        DeviceIO()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# HumanInLoopMixin：不可脚本化操作返回 InterventionRequest 而非抛错（需求 14.2）
# ---------------------------------------------------------------------------
class _HalfAutoLike(HumanInLoopMixin, DeviceIO):
    """复用 HumanInLoopMixin 的半自动后端样例：仅 open_serial/capture/send_cmd
    可脚本化，build/flash/inject 由 mixin 降级为干预点。"""

    def __init__(self, cfg: DeviceConfig) -> None:
        self.cfg = cfg

    def open_serial(self, port: str, baud: int) -> None:
        return None

    def capture(self, duration_s: float) -> RawTraceArtifact:  # pragma: no cover
        raise NotImplementedError

    def send_cmd(self, cmd: str) -> str:
        return "resp"


def test_mixin_build_returns_intervention_request():
    device = _HalfAutoLike(_make_cfg("half_auto"))
    result = device.build()
    assert isinstance(result, InterventionRequest)
    assert result.action == "build"
    assert result.instruction  # 非空可读指引


def test_mixin_flash_returns_intervention_request():
    device = _HalfAutoLike(_make_cfg("half_auto"))
    result = device.flash()
    assert isinstance(result, InterventionRequest)
    assert result.action == "flash"


def test_mixin_inject_input_returns_intervention_request_with_event_context():
    device = _HalfAutoLike(_make_cfg("half_auto"))
    event = InputEvent(kind="touch", params={"x": 10, "y": 20})
    result = device.inject_input(event)
    assert isinstance(result, InterventionRequest)
    assert result.action == "inject_input"
    # 干预请求应携带待注入事件供操作者参考
    assert result.context["event"]["kind"] == "touch"
    assert result.context["event"]["params"] == {"x": 10, "y": 20}


def test_mixin_does_not_raise():
    """人在环降级绝不抛错——这是 Half_Auto_Mode 的前提（需求 14.2）。"""
    device = _HalfAutoLike(_make_cfg("half_auto"))
    device.build()
    device.flash()
    device.inject_input(InputEvent(kind="key", params={"code": 13}))


# ---------------------------------------------------------------------------
# 辅助数据类型：可构造 + JSON round-trip
# ---------------------------------------------------------------------------
def test_build_result_round_trip():
    br = BuildResult(success=False, output="log", error="compile error")
    assert BuildResult.model_validate_json(br.model_dump_json()) == br


def test_flash_result_round_trip():
    fr = FlashResult(success=True, output="flashed")
    assert FlashResult.model_validate_json(fr.model_dump_json()) == fr


def test_input_event_round_trip():
    ev = InputEvent(kind="gesture", params={"dir": "up", "len": 100})
    assert InputEvent.model_validate_json(ev.model_dump_json()) == ev


def test_intervention_request_round_trip():
    ir = InterventionRequest(
        action="flash", instruction="请手动烧录", context={"cmd": "make flash"}
    )
    assert InterventionRequest.model_validate_json(ir.model_dump_json()) == ir


def test_defaults_are_sane():
    assert BuildResult(success=True).output == ""
    assert FlashResult(success=True).error is None
    assert InputEvent(kind="touch").params == {}
    assert InterventionRequest(action="build", instruction="x").context == {}
