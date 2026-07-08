"""Collector 子智能体：指挥 DeviceIO 采集 LVGL profiler trace（任务 11.2）。

严格对照 design.md「编排层与智能体层」中 Collector 的职责，以及需求 2：

| 子智能体 | 职责 | 依赖 |
|----------|------|------|
| Collector | 指挥采集、判断日志完整性、必要时降波特率重采 | DeviceIO, Trace_Filter |

Collector 是**采集阶段**的推理主体，但其一切副作用都经注入的 ``DeviceIO`` 接口完成
（打开串口、注入输入、采集日志），并借助确定性工具 ``trace_filter`` 判断日志完整性。
因此测试时以 ``FakeDeviceIO`` 替换即可脱离真实硬件驱动完整采集流程（依赖注入友好）。

行为（严格对照需求 2）：

* **需求 2.2**：串口连接打开且场景被触发后，把原始 LVGL profiler trace 采集到一个与
  该 Tuning_Run 关联的原始日志工件中（``capture`` 落盘，Collector 只持 ``ArtifactRef``）。
* **需求 2.5**：在需要输入注入才能触发场景时，于采集期间注入所配置的触摸/按键输入
  以触发执行该场景（``inject_input``）。
* **需求 2.6**：采集完成时，在运行状态中记录**采集时长**与**原始日志工件位置**
  （见 :meth:`Collector.to_state_update`：仅写 ``ArtifactRef``，绝不内联日志内容——
  Property 12）。

**降波特率重采**（design.md「Error Handling」：日志线程交错破坏 B/E 配对时，Collector
可决定降波特率重采）：采集后用 ``trace_filter`` 判断完整性；若因串口误码/交错导致事件
被排除或出现受损区间，则将波特率**减半**（天然 ``<= max_safe_baud``，需求 2.3），
重新打开串口并重采，直至日志完整、达到最大重试次数、或波特率触及下限。

关键不变量（Property 12：原始日志不进上下文）：Collector 在运行状态中只记录指向落盘
工件的 ``ArtifactRef``，原始日志 / systrace 内容绝不进入 state 或 LLM context。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from embedded_device_agent.capabilities.perf_tuning.tools.trace_filter import (
    trace_filter,
)
from embedded_device_agent.core.config.models import DeviceConfig
from embedded_device_agent.core.device.base import DeviceIO
from embedded_device_agent.core.device.models import InputEvent
from embedded_device_agent.core.models import (
    ArtifactRef,
    FilterResult,
    RawTraceArtifact,
)

__all__ = ["Collector", "CollectResult"]

# 降波特率重采时的默认下限：低于此值再降已无实际收益，用于保证重采循环有界终止。
_DEFAULT_MIN_BAUD = 9600


class CollectResult(BaseModel):
    """一次采集（含可能的降波特率重采）的结构化结果。

    以结果对象建模（而非裸异常），便于确定性断言与后续写入运行状态。字段全部
    可无损 JSON 序列化（Property 9）；``artifact`` 仅为指向落盘原始日志的引用，
    不含日志内容（Property 12）。
    """

    artifact: ArtifactRef  # 原始日志工件位置（需求 2.6）
    duration_s: float  # 采集时长（需求 2.6）
    baud: int  # 最终成功采集所用波特率（可能已降）
    attempts: int  # 实际采集尝试次数（含重采）
    complete: bool  # 日志是否判定为完整
    retained: int  # trace_filter 保留的事件数
    excluded: int  # trace_filter 排除的事件数
    corrupted_regions: list[tuple[int, int]] = Field(default_factory=list)


class Collector:
    """采集阶段子智能体：指挥 DeviceIO 采集并判断日志完整性（需求 2）。

    副作用经注入的 :class:`DeviceIO` 完成，完整性判断经注入的 ``filter_fn``
    （默认确定性工具 :func:`trace_filter`）完成——两者皆可在测试中替换为
    fake，从而脱离真实硬件与文件系统副作用离线自测。

    Parameters
    ----------
    device:
        设备控制面抽象；用于 ``open_serial`` / ``inject_input`` / ``capture``。
    config:
        设备配置，提供 ``port`` / ``baud`` / ``max_safe_baud``。
    filter_fn:
        判断日志完整性所用的清洗函数；缺省为 :func:`trace_filter`。签名需与
        ``trace_filter(raw) -> FilterResult`` 一致。
    max_recapture:
        因串口误码/交错导致日志不完整时，允许的**额外**降波特率重采次数上限
        （不含首次采集）；用于保证重采循环有界终止。
    min_baud:
        降波特率的下限；低于此值不再继续降速重采。
    """

    def __init__(
        self,
        device: DeviceIO,
        config: DeviceConfig,
        *,
        filter_fn: Callable[[RawTraceArtifact], FilterResult] = trace_filter,
        max_recapture: int = 3,
        min_baud: int = _DEFAULT_MIN_BAUD,
    ) -> None:
        self._device = device
        self._config = config
        self._filter_fn = filter_fn
        self._max_recapture = max_recapture
        self._min_baud = min_baud

    def collect(
        self,
        *,
        run_id: str,
        duration_s: float,
        trigger_event: InputEvent | None = None,
        port: str | None = None,
        baud: int | None = None,
    ) -> CollectResult:
        """采集某场景的原始 profiler trace，必要时降波特率重采（需求 2.2/2.5/2.6）。

        Parameters
        ----------
        run_id:
            当前 Tuning_Run 标识；产出的 ``ArtifactRef`` 据此与该运行关联（需求 2.2）。
        duration_s:
            单次采集时长（秒）。
        trigger_event:
            需要输入注入才能触发场景时提供；采集前经 ``inject_input`` 注入以触发
            执行该场景（需求 2.5）。为 ``None`` 时表示场景无需注入即可产生 trace。
        port / baud:
            覆盖配置中的端口/起始波特率（缺省取 ``config.port`` / ``config.baud``）。

        Returns
        -------
        CollectResult
            成功采集的原始日志工件引用、采集时长、最终波特率、尝试次数与完整性判定。

        Notes
        -----
        降波特率重采：若首次采集的日志因交错/误码不完整，则将波特率减半后重开串口
        重采（天然满足 ``<= max_safe_baud``，需求 2.3），直至完整、达到
        ``max_recapture`` 次重采、或波特率触及 ``min_baud``。返回最后一次采集结果
        （即便仍不完整也如实返回，由编排层按自动化模式决定下一步）。
        """
        port = port if port is not None else self._config.port
        current_baud = baud if baud is not None else self._config.baud

        attempts = 0
        result: CollectResult | None = None

        while True:
            attempts += 1
            # 需求 2.1（经 DeviceIO 落实）：以配置端口/波特率打开串口连接。
            self._device.open_serial(port, current_baud)

            # 需求 2.5：需要注入才能触发的场景，于采集期间注入触摸/按键输入。
            if trigger_event is not None:
                self._device.inject_input(trigger_event)

            # 需求 2.2：把原始 LVGL profiler trace 采集到与本 run 关联的原始日志工件。
            raw = self._device.capture(duration_s)

            # 借确定性 trace_filter 判断完整性（不改动原始工件，仅读取判定）。
            filter_result = self._filter_fn(raw)
            complete = self._is_complete(filter_result)

            result = CollectResult(
                artifact=self._to_artifact_ref(run_id, raw),
                duration_s=raw.duration_s,
                baud=current_baud,
                attempts=attempts,
                complete=complete,
                retained=filter_result.retained,
                excluded=filter_result.excluded,
                corrupted_regions=list(filter_result.corrupted_regions),
            )

            if complete:
                break
            # 不完整：尝试降波特率重采（需求 3.2 Error Handling / 需求 2.3）。
            next_baud = current_baud // 2
            if attempts > self._max_recapture or next_baud < self._min_baud:
                break
            current_baud = next_baud

        return result

    @staticmethod
    def _is_complete(filter_result: FilterResult) -> bool:
        """判断日志完整性：无事件被排除、无受损区间，且确有事件被保留。

        串口误码或其他线程交错会破坏 B/E 配对，被 ``trace_filter`` 记为 ``excluded``
        或 ``corrupted_regions``（需求 3.2）；据此判定不完整并触发降波特率重采。
        零保留（未采到任何事件）亦视为不完整。
        """
        return (
            filter_result.retained > 0
            and filter_result.excluded == 0
            and not filter_result.corrupted_regions
        )

    @staticmethod
    def _to_artifact_ref(run_id: str, raw: RawTraceArtifact) -> ArtifactRef:
        """由落盘原始日志构造轻量 ``ArtifactRef``（仅引用，不含内容，Property 12）。"""
        raw_path = Path(raw.path)
        try:
            size_bytes = raw_path.stat().st_size
        except OSError:
            # 采集工件尚未落盘（如 fake 回放的占位路径）时，体量记 0，不影响引用语义。
            size_bytes = 0
        return ArtifactRef(
            run_id=run_id,
            kind="raw_log",
            path=raw_path,
            size_bytes=size_bytes,
        )

    @staticmethod
    def to_state_update(result: CollectResult) -> dict[str, object]:
        """把采集结果落为运行状态的增量更新（需求 2.6）。

        遵循 LangGraph 节点「返回状态增量」的惯例：把**原始日志工件位置**写入
        ``latest_artifact``。运行状态只承载 ``ArtifactRef`` 引用，绝不内联日志内容
        （Property 12）；采集时长随 ``CollectResult`` / 工件（``RawTraceArtifact``）
        一并留存，供编排层记录与后续对比。
        """
        return {"latest_artifact": result.artifact}
