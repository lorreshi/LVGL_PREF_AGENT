"""确定性 Test_Harness 的 fake 实现：``FakeDeviceIO`` 与 ``FakeLLMProvider``。

对照 design.md "Test_Harness" 章节与需求 20：

* ``FakeDeviceIO`` 实现 ``DeviceIO`` 接口，回放**录制的采集日志**与**预置的
  构建/烧录结果**，不触达真实串口硬件（需求 20.1）。
* ``FakeLLMProvider`` 实现 ``BaseLLMProvider`` 接口，返回一个回放**预置决策**的
  LangChain chat model，不调用真实 LLM API（需求 20.2）。

两者均为纯回放、确定性：给定同样的构造入参，多次调用产出完全一致的结果，使任意
能力仅经注入这些 fake 即可离线、可重复地驱动完整 subgraph（需求 20.3 / 20.4）。
所有可回放序列在耗尽后重复最后一项，从而对"调用次数超出录制"这一情况保持稳健。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from embedded_device_agent.core.device.base import DeviceIO
from embedded_device_agent.core.device.models import (
    BuildResult,
    FlashResult,
    InputEvent,
)
from embedded_device_agent.core.llm.base import BaseLLMProvider
from embedded_device_agent.core.models import RawTraceArtifact

# 兼容以模块或包两种方式导入 fixtures 路径常量。
try:  # pragma: no cover - 导入路径兼容
    from tests.fixtures import RECORDED_CAPTURE_LOG
except Exception:  # pragma: no cover
    RECORDED_CAPTURE_LOG = (
        Path(__file__).resolve().parent.parent / "fixtures" / "recorded_capture.log"
    )

__all__ = ["FakeDeviceIO", "FakeLLMProvider"]

#: ``FakeDeviceIO`` 未显式提供预置结果时使用的确定性默认值。
_DEFAULT_BUILD_RESULT = BuildResult(success=True, output="fake build ok")
_DEFAULT_FLASH_RESULT = FlashResult(success=True, output="fake flash ok")
#: 确定性的采集时间戳（不使用真实时钟，保证可重复）。
_FIXED_CAPTURED_AT = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


class FakeDeviceIO(DeviceIO):
    """回放录制日志与预置结果的确定性 ``DeviceIO``（需求 20.1）。

    不打开任何真实串口、不执行任何子进程；``capture`` 返回指向**录制日志 fixture**
    的 ``RawTraceArtifact``，``build`` / ``flash`` 依序回放预置结果。所有调用被记录
    到 ``calls`` 供测试断言。

    参数
    ----
    capture_log:
        录制采集日志的路径；``capture`` 返回的工件即指向该文件。默认使用
        ``tests/fixtures/recorded_capture.log``。
    build_results / flash_results:
        预置的构建/烧录结果序列，按调用次序回放；耗尽后重复最后一项。
        省略时使用一个成功的默认结果。
    cmd_responses:
        ``send_cmd`` 的预置响应序列，按调用次序回放；耗尽后重复最后一项。
    baud:
        录制日志采集时的波特率，写入 ``RawTraceArtifact.baud``。
    """

    def __init__(
        self,
        *,
        capture_log: str | Path | None = None,
        build_results: Sequence[BuildResult] | None = None,
        flash_results: Sequence[FlashResult] | None = None,
        cmd_responses: Sequence[str] | None = None,
        baud: int = 115200,
        run_id_prefix: str = "fake-run",
    ) -> None:
        self._capture_log = Path(capture_log) if capture_log else RECORDED_CAPTURE_LOG
        self._build_results = list(build_results) if build_results else [_DEFAULT_BUILD_RESULT]
        self._flash_results = list(flash_results) if flash_results else [_DEFAULT_FLASH_RESULT]
        self._cmd_responses = list(cmd_responses) if cmd_responses else ["OK"]
        self._baud = baud
        self._run_id_prefix = run_id_prefix

        # 回放游标与调用记录（确定性、可断言）。
        self._capture_count = 0
        self._build_count = 0
        self._flash_count = 0
        self._cmd_count = 0
        self.calls: list[tuple[str, object]] = []
        self.injected_inputs: list[InputEvent] = []
        self.opened: tuple[str, int] | None = None

    # -- 串口开合（纯记录，不触达硬件）-------------------------------------
    def open_serial(self, port: str, baud: int) -> None:
        """记录一次串口打开请求，但不打开任何真实串口。"""
        self.opened = (port, baud)
        self.calls.append(("open_serial", (port, baud)))

    # -- 采集：回放录制日志 -------------------------------------------------
    def capture(self, duration_s: float) -> RawTraceArtifact:
        """返回指向录制日志 fixture 的 ``RawTraceArtifact``（需求 20.1）。"""
        run_id = f"{self._run_id_prefix}-{self._capture_count:04d}"
        self._capture_count += 1
        self.calls.append(("capture", duration_s))
        return RawTraceArtifact(
            run_id=run_id,
            path=self._capture_log,
            captured_at=_FIXED_CAPTURED_AT,
            duration_s=duration_s,
            baud=self._baud,
        )

    # -- 构建 / 烧录：回放预置结果 -----------------------------------------
    def build(self) -> BuildResult:
        """回放下一个预置 ``BuildResult``（耗尽后重复最后一项）。"""
        result = self._next(self._build_results, self._build_count)
        self._build_count += 1
        self.calls.append(("build", None))
        return result

    def flash(self) -> FlashResult:
        """回放下一个预置 ``FlashResult``（耗尽后重复最后一项）。"""
        result = self._next(self._flash_results, self._flash_count)
        self._flash_count += 1
        self.calls.append(("flash", None))
        return result

    # -- 命令 / 输入（纯记录 / 回放）---------------------------------------
    def send_cmd(self, cmd: str) -> str:
        """回放下一个预置命令响应（耗尽后重复最后一项）。"""
        resp = self._next(self._cmd_responses, self._cmd_count)
        self._cmd_count += 1
        self.calls.append(("send_cmd", cmd))
        return resp

    def inject_input(self, event: InputEvent) -> None:
        """记录一次注入事件，但不触达真实设备（需求 20.1 精神）。"""
        self.injected_inputs.append(event)
        self.calls.append(("inject_input", event))

    # -- 内部辅助 -----------------------------------------------------------
    @staticmethod
    def _next(seq, index):
        """取序列第 ``index`` 项；越界则重复最后一项（保证稳健与确定性）。"""
        if index < len(seq):
            return seq[index]
        return seq[-1]


class FakeLLMProvider(BaseLLMProvider):
    """回放预置决策的确定性 ``BaseLLMProvider``（需求 20.2）。

    ``get_chat_model`` 返回一个 ``FakeListChatModel``，按构造时给定的
    ``responses`` 依序回放预置文本决策，不发起任何真实 LLM API 调用。

    参数
    ----
    responses:
        预置的模型输出序列，按调用次序回放。至少需一项。
    name:
        该 provider 的可读标识，默认 ``"fake"``。
    """

    def __init__(self, responses: Sequence[str], *, name: str = "fake") -> None:
        if not responses:
            raise ValueError("FakeLLMProvider 需要至少一个预置 response 用于回放。")
        self._responses = list(responses)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_chat_model(self) -> BaseChatModel:
        """返回回放预置决策的 ``FakeListChatModel``，不调用真实 LLM。"""
        return FakeListChatModel(responses=list(self._responses))
