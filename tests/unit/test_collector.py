"""任务 11.3：Collector 子智能体单元测试（需求 2.2 / 2.5 / 2.6）。

以确定性 ``FakeDeviceIO``（任务 16.1）替换真实设备控制面，脱离串口硬件离线驱动
Collector 的完整采集流程，验证三条职责：

* **需求 2.2**：串口打开、场景触发后把原始 trace 采集为与本 run 关联的原始日志
  工件，Collector 仅持指向落盘工件的 ``ArtifactRef``。
* **需求 2.5**：需要输入注入触发的场景，采集前经 ``inject_input`` 注入所配置事件。
* **需求 2.6**：采集完成时在运行状态中记录采集时长与原始日志工件位置
  （``to_state_update`` 仅写 ``latest_artifact``，Property 12）。

外加降波特率重采决策（design.md「Error Handling」/ 需求 2.3）：首次日志被判不完整
时将波特率减半重采，天然满足 ``<= max_safe_baud``；有界终止于完整 / 触及重采上限 /
触及波特率下限。由于 ``FakeDeviceIO`` 每次 ``capture`` 都回放同一录制日志，「先不
完整、后完整」的完整性演化经注入自定义 ``filter_fn`` 确定性驱动。
"""

from __future__ import annotations

from pathlib import Path

from embedded_device_agent.capabilities.perf_tuning.agents.collector import (
    CollectResult,
    Collector,
)
from embedded_device_agent.core.config.models import DeviceConfig
from embedded_device_agent.core.device.models import InputEvent
from embedded_device_agent.core.models import (
    ArtifactRef,
    FilterResult,
    RawTraceArtifact,
)
from tests.fixtures import RECORDED_CAPTURE_LOG
from tests.harness import FakeDeviceIO


# ---------------------------------------------------------------------------
# 测试脚手架
# ---------------------------------------------------------------------------
def _config(*, baud: int = 921600, max_safe_baud: int = 921600) -> DeviceConfig:
    """构造一份可脚本化的 fake 设备配置。"""
    return DeviceConfig(
        type="fake",
        port="/dev/ttyFAKE0",
        baud=baud,
        max_safe_baud=max_safe_baud,
    )


def _complete_result() -> FilterResult:
    """一个「日志完整」判定：有事件被保留、零排除、无受损区间。"""
    return FilterResult(
        clean_path=Path("/tmp/clean.systrace"),
        retained=12,
        excluded=0,
        corrupted_regions=[],
    )


def _incomplete_result() -> FilterResult:
    """一个「日志不完整」判定：存在被排除事件与受损区间（交错破坏 B/E 配对）。"""
    return FilterResult(
        clean_path=Path("/tmp/clean.systrace"),
        retained=8,
        excluded=3,
        corrupted_regions=[(11, 12)],
    )


class _ScriptedFilter:
    """按预置序列依次回放 ``FilterResult`` 的确定性 ``filter_fn``（耗尽后重复末项）。

    ``FakeDeviceIO`` 每次 ``capture`` 都回放同一录制日志，无法表达「先不完整、后
    完整」的演化；故经此注入的完整性判定序列来确定性驱动降波特率重采决策，并记录
    每次判定所依据的原始工件用于断言。
    """

    def __init__(self, results: list[FilterResult]) -> None:
        self._results = results
        self.seen: list[RawTraceArtifact] = []

    def __call__(self, raw: RawTraceArtifact) -> FilterResult:
        self.seen.append(raw)
        idx = min(len(self.seen) - 1, len(self._results) - 1)
        return self._results[idx]


# ---------------------------------------------------------------------------
# 需求 2.2 / 2.6：正常采集流程产出 ArtifactRef 并记录时长与工件位置
# ---------------------------------------------------------------------------
def test_normal_capture_produces_artifact_ref_and_records_duration_location() -> None:
    dev = FakeDeviceIO()  # 默认回放完整录制日志
    collector = Collector(dev, _config(), filter_fn=lambda raw: _complete_result())

    result = collector.collect(run_id="run-alpha", duration_s=1.5)

    assert isinstance(result, CollectResult)
    assert result.complete is True
    assert result.attempts == 1  # 完整 → 无需重采

    # 需求 2.2：产出与本 run 关联、指向落盘原始日志的 ArtifactRef（仅引用，不含内容）。
    assert isinstance(result.artifact, ArtifactRef)
    assert result.artifact.run_id == "run-alpha"
    assert result.artifact.kind == "raw_log"
    assert result.artifact.path == RECORDED_CAPTURE_LOG
    assert result.artifact.size_bytes > 0  # 真实落盘 fixture 体量已记录

    # 需求 2.6：记录采集时长（随录制工件回放的 duration_s）。
    assert result.duration_s == 1.5

    # 经注入 DeviceIO 完成副作用：以配置端口 / 起始波特率打开串口、采集一次。
    assert dev.opened == ("/dev/ttyFAKE0", 921600)
    assert dev.calls.count(("capture", 1.5)) == 1


def test_normal_capture_without_trigger_does_not_inject_input() -> None:
    dev = FakeDeviceIO()
    collector = Collector(dev, _config(), filter_fn=lambda raw: _complete_result())

    collector.collect(run_id="run-noinput", duration_s=1.0)

    # trigger_event 缺省 → 不应注入任何输入。
    assert dev.injected_inputs == []
    assert all(call[0] != "inject_input" for call in dev.calls)


def test_collect_honors_port_and_baud_overrides() -> None:
    dev = FakeDeviceIO()
    collector = Collector(dev, _config(), filter_fn=lambda raw: _complete_result())

    collector.collect(
        run_id="run-override",
        duration_s=1.0,
        port="/dev/ttyOTHER",
        baud=230400,
    )

    assert dev.opened == ("/dev/ttyOTHER", 230400)


# ---------------------------------------------------------------------------
# 需求 2.5：需要输入注入触发的场景，采集前注入所配置事件
# ---------------------------------------------------------------------------
def test_input_injection_triggers_scenario_before_capture() -> None:
    dev = FakeDeviceIO()
    collector = Collector(dev, _config(), filter_fn=lambda raw: _complete_result())
    trigger = InputEvent(kind="touch", params={"x": 100, "y": 200})

    collector.collect(run_id="run-inject", duration_s=2.0, trigger_event=trigger)

    # 需求 2.5：所配置的触摸/按键输入经 DeviceIO 注入以触发场景。
    assert dev.injected_inputs == [trigger]
    assert ("inject_input", trigger) in dev.calls

    # 注入须发生在采集之前（先触发场景，再采集其 trace）。
    kinds = [name for name, _ in dev.calls]
    assert kinds.index("inject_input") < kinds.index("capture")


# ---------------------------------------------------------------------------
# 降波特率重采决策（需求 2.3 / design.md Error Handling）
# ---------------------------------------------------------------------------
def test_baud_reduction_recapture_when_first_log_incomplete() -> None:
    dev = FakeDeviceIO()
    # 首次判定不完整（交错破坏配对）→ 降波特率重采；第二次判定完整 → 停止。
    scripted = _ScriptedFilter([_incomplete_result(), _complete_result()])
    collector = Collector(dev, _config(baud=921600, max_safe_baud=921600), filter_fn=scripted)

    result = collector.collect(run_id="run-recapture", duration_s=1.0)

    assert result.attempts == 2  # 首采 + 一次降波特率重采
    assert result.complete is True
    # 波特率减半：921600 → 460800，天然 <= max_safe_baud（需求 2.3）。
    assert result.baud == 460800
    assert dev.opened == ("/dev/ttyFAKE0", 460800)

    # 两次采集分别以原始波特率与减半后波特率打开串口。
    open_bauds = [args[1] for name, args in dev.calls if name == "open_serial"]
    assert open_bauds == [921600, 460800]
    assert dev.calls.count(("capture", 1.0)) == 2


def test_recapture_bauds_never_exceed_max_safe_baud() -> None:
    dev = FakeDeviceIO()
    scripted = _ScriptedFilter([_incomplete_result(), _incomplete_result(), _complete_result()])
    collector = Collector(dev, _config(baud=921600, max_safe_baud=921600), filter_fn=scripted)

    collector.collect(run_id="run-safe", duration_s=1.0)

    open_bauds = [args[1] for name, args in dev.calls if name == "open_serial"]
    assert open_bauds == [921600, 460800, 230400]
    assert all(b <= 921600 for b in open_bauds)  # 需求 2.3：始终 <= max_safe_baud


def test_recapture_is_bounded_and_returns_last_result_when_never_complete() -> None:
    dev = FakeDeviceIO()
    # 始终不完整 → 采集次数受 max_recapture 上界约束，返回最后一次（仍不完整）结果。
    scripted = _ScriptedFilter([_incomplete_result()])
    collector = Collector(
        dev,
        _config(baud=921600, max_safe_baud=921600),
        filter_fn=scripted,
        max_recapture=2,
        min_baud=9600,
    )

    result = collector.collect(run_id="run-bounded", duration_s=1.0)

    # 首采 + 至多 max_recapture 次重采 = 3 次尝试后有界终止。
    assert result.attempts == 3
    assert result.complete is False
    assert result.excluded == 3  # 如实反映不完整判定，交由编排层决定下一步


def test_recapture_stops_at_min_baud_floor() -> None:
    dev = FakeDeviceIO()
    scripted = _ScriptedFilter([_incomplete_result()])
    # 起始波特率减半即触及下限：38400 // 2 = 19200 < min_baud=20000 → 不再降速重采。
    collector = Collector(
        dev,
        _config(baud=38400, max_safe_baud=38400),
        filter_fn=scripted,
        max_recapture=5,
        min_baud=20000,
    )

    result = collector.collect(run_id="run-floor", duration_s=1.0)

    # 首采不完整，减半后低于下限 → 停止，仅一次尝试。
    assert result.attempts == 1
    assert result.baud == 38400


# ---------------------------------------------------------------------------
# 需求 2.6 / Property 12：to_state_update 仅写 latest_artifact（ArtifactRef 引用）
# ---------------------------------------------------------------------------
def test_to_state_update_writes_only_latest_artifact_ref() -> None:
    dev = FakeDeviceIO()
    collector = Collector(dev, _config(), filter_fn=lambda raw: _complete_result())
    result = collector.collect(run_id="run-state", duration_s=1.0)

    update = Collector.to_state_update(result)

    # 运行状态增量只承载指向落盘工件的引用，绝不内联原始日志内容（Property 12）。
    assert set(update.keys()) == {"latest_artifact"}
    assert update["latest_artifact"] is result.artifact
    assert isinstance(update["latest_artifact"], ArtifactRef)
    assert update["latest_artifact"].kind == "raw_log"
    # 更新可无损 JSON 序列化（Property 9），且不含任何日志正文字段。
    dumped = ArtifactRef.model_validate(update["latest_artifact"].model_dump())
    assert dumped.run_id == "run-state"
