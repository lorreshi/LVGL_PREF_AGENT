"""Instrumentor（埋点器）子智能体：在 C 源码中下钻插入更细粒度 profiler 埋点。

对照 design.md「编排层与智能体层 → Instrumentor」与需求 6：当排序后的热点归结为
一个内部耗时分布未知的**粗粒度函数**时，需要在其内部插入成对的
``LV_PROFILER_BEGIN_TAG`` / ``LV_PROFILER_END_TAG`` 埋点，重编译烧录后针对同一场景
复测，从而把耗时进一步归因到叶子级热点。

职责边界（保持可测试）：

* **源码编辑逻辑为纯函数**——给定源码字符串与埋点点位，恒产出确定的新源码，不触碰
  磁盘、不产生任何副作用（便于离线单测）。见 :func:`insert_instrumentation` /
  :func:`insert_many` / :func:`render_begin_tag` / :func:`render_end_tag`。
* **副作用藏在注入接口之后**——重编译与烧录经注入的 :class:`DeviceIO` 完成
  （需求 6.3）；可选的落盘写入经注入的 ``write`` 回调完成。推理（决定埋点位置）经
  注入的 :class:`BaseLLMProvider` 完成，与确定性编辑分离。

关键不变量：

* **成对性（需求 6.1）**：每插入一处埋点必成对产出 BEGIN/END，且二者标签一致；
  区域以 ``[begin_line, end_line]`` 描述，BEGIN 插到区域首行之前、END 插到区域末行
  之后，天然保证嵌套配对。
* **字符串字面量标签（需求 6.2）**：标签参数一律经 :func:`c_string_literal` 转义并
  以 C 字符串字面量形式输出，杜绝把标签当作裸标识符。
* **精度守卫（需求 6.6）**：当默认时间戳精度比区分下钻标签所需的亚毫秒分辨率更粗
  时，报告"需先引入微秒级 tick 回调，细粒度计时方可信"。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from pydantic import BaseModel, Field, model_validator

from embedded_device_agent.core.device.base import DeviceIO
from embedded_device_agent.core.device.models import (
    BuildResult,
    FlashResult,
    InterventionRequest,
)
from embedded_device_agent.core.llm.base import BaseLLMProvider

__all__ = [
    "SUBMS_RESOLUTION_US",
    "InstrumentationPoint",
    "PrecisionReport",
    "InstrumentationResult",
    "c_string_literal",
    "render_begin_tag",
    "render_end_tag",
    "insert_instrumentation",
    "insert_many",
    "check_timing_precision",
    "Instrumentor",
]

# 亚毫秒分辨率阈值（微秒）：区分下钻标签至少需要优于 1ms 的计时精度（需求 6.6）。
SUBMS_RESOLUTION_US = 1_000

# LVGL profiler 埋点宏名（对照需求 6.1）。
_BEGIN_MACRO = "LV_PROFILER_BEGIN_TAG"
_END_MACRO = "LV_PROFILER_END_TAG"


class InstrumentationPoint(BaseModel):
    """一处待插入的成对埋点：把 C 源文件 ``[begin_line, end_line]`` 区域包进 BEGIN/END。

    行号为 **1 基**，闭区间语义：``LV_PROFILER_BEGIN_TAG`` 插入到 ``begin_line`` 之前，
    ``LV_PROFILER_END_TAG`` 插入到 ``end_line`` 之后；二者共用同一 ``tag``，保证配对
    （需求 6.1）。``file`` 仅作记录/定位用，纯编辑函数不据此读写磁盘。
    """

    file: str
    tag: str
    begin_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_span(self) -> "InstrumentationPoint":
        if self.end_line < self.begin_line:
            raise ValueError(
                f"埋点区域非法：end_line({self.end_line}) < begin_line({self.begin_line})"
            )
        if not self.tag.strip():
            raise ValueError("埋点标签不能为空")
        return self


class PrecisionReport(BaseModel):
    """时间戳精度评估结果（需求 6.6）。

    ``sufficient`` 为真表示默认时间戳精度足以区分下钻标签；为假时 ``message`` 给出
    "需微秒级 tick 回调后细粒度计时方可信"的可读报告，供 Orchestrator 转达用户。
    """

    sufficient: bool
    tick_resolution_us: int
    required_resolution_us: int
    message: str


class InstrumentationResult(BaseModel):
    """一次下钻埋点的结构化结果。

    * ``edited_sources``：``file -> 新源码`` 的映射（纯编辑产物）。
    * ``points`` / ``tags``：本次插入的点位与标签（标签去重、保序）。
    * ``precision``：时间戳精度评估（需求 6.6）。
    * ``build`` / ``flash``：经 DeviceIO 触发的重编译/烧录结果；半自动后端可能返回
      :class:`InterventionRequest`（人工烧录）。未触发时为 ``None``。
    * ``reflashed``：是否已成功自动重编译并烧录。
    """

    edited_sources: dict[str, str] = Field(default_factory=dict)
    points: list[InstrumentationPoint] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    precision: PrecisionReport
    build: BuildResult | InterventionRequest | None = None
    flash: FlashResult | InterventionRequest | None = None
    reflashed: bool = False


# --------------------------------------------------------------------------- #
# 纯函数：埋点渲染与源码编辑（无副作用，确定性，可离线单测）
# --------------------------------------------------------------------------- #
def c_string_literal(tag: str) -> str:
    """把标签名转义为 C 字符串字面量（含首尾双引号），满足需求 6.2。

    转义反斜杠、双引号与常见控制字符，避免把标签误当作裸标识符或破坏源码。
    """
    escaped = (
        tag.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def render_begin_tag(tag: str, *, indent: str = "") -> str:
    """渲染一行 ``LV_PROFILER_BEGIN_TAG("tag");``（标签为字符串字面量）。"""
    return f"{indent}{_BEGIN_MACRO}({c_string_literal(tag)});"


def render_end_tag(tag: str, *, indent: str = "") -> str:
    """渲染一行 ``LV_PROFILER_END_TAG("tag");``（标签为字符串字面量）。"""
    return f"{indent}{_END_MACRO}({c_string_literal(tag)});"


def _leading_indent(line: str) -> str:
    """提取某行的前导空白，用以让插入的埋点与目标区域缩进对齐。"""
    return line[: len(line) - len(line.lstrip())]


def insert_instrumentation(source: str, point: InstrumentationPoint) -> str:
    """在 ``source`` 的 ``point`` 区域外成对插入 BEGIN/END 埋点，返回新源码（纯函数）。

    行号为 1 基闭区间；``begin_line`` / ``end_line`` 必须落在源码行范围内，否则报错。
    插入的埋点沿用目标行缩进，保证嵌套配对与可读性（需求 6.1 / 6.2）。
    """
    lines = source.splitlines(keepends=True)
    n = len(lines)
    if point.begin_line > n or point.end_line > n:
        raise ValueError(
            f"埋点行号越界：源码共 {n} 行，收到 begin={point.begin_line} end={point.end_line}"
        )

    begin_idx = point.begin_line - 1
    end_idx = point.end_line - 1
    begin_indent = _leading_indent(lines[begin_idx])
    end_indent = _leading_indent(lines[end_idx])

    # 保持行内换行风格：若目标行以换行结尾则沿用，否则退化为 "\n"。
    def _nl(idx: int) -> str:
        line = lines[idx]
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
        return "\n"

    begin_line = render_begin_tag(point.tag, indent=begin_indent) + _nl(begin_idx)
    end_line = render_end_tag(point.tag, indent=end_indent) + _nl(end_idx)

    # 先插 END（在末行之后），再插 BEGIN（在首行之前），避免先插 BEGIN 打乱 end_idx。
    new_lines = list(lines)
    new_lines.insert(end_idx + 1, end_line)
    new_lines.insert(begin_idx, begin_line)
    return "".join(new_lines)


def insert_many(source: str, points: Iterable[InstrumentationPoint]) -> str:
    """在同一份源码中插入多处埋点（纯函数）。

    按区域**从后往前**依次插入，使靠前区域的行号在插入靠后区域时仍然有效，避免行号
    漂移。区域之间应互不重叠（重叠区域的配对语义未定义，调用方需保证）。
    """
    ordered = sorted(points, key=lambda p: (p.begin_line, p.end_line), reverse=True)
    result = source
    for point in ordered:
        result = insert_instrumentation(result, point)
    return result


def check_timing_precision(
    tick_resolution_us: int, *, required_resolution_us: int = SUBMS_RESOLUTION_US
) -> PrecisionReport:
    """评估默认时间戳精度是否足以区分下钻标签（需求 6.6）。

    当 ``tick_resolution_us``（默认 tick 的最小分辨率）**不优于**所需的亚毫秒分辨率
    （即 ``>= required_resolution_us``）时，判定精度不足，报告需引入微秒级 tick 回调
    后细粒度计时方可信。
    """
    sufficient = tick_resolution_us < required_resolution_us
    if sufficient:
        message = (
            f"默认时间戳精度 {tick_resolution_us}us 优于所需的 "
            f"{required_resolution_us}us 亚毫秒分辨率，下钻标签计时可信。"
        )
    else:
        message = (
            f"默认时间戳精度 {tick_resolution_us}us 比区分下钻标签所需的 "
            f"{required_resolution_us}us 亚毫秒分辨率更粗；在引入微秒级精度的 tick "
            f"回调之前，更细粒度的埋点计时不可信。"
        )
    return PrecisionReport(
        sufficient=sufficient,
        tick_resolution_us=tick_resolution_us,
        required_resolution_us=required_resolution_us,
        message=message,
    )


# --------------------------------------------------------------------------- #
# Instrumentor 子智能体：把纯编辑与注入的副作用/推理接口编排在一起
# --------------------------------------------------------------------------- #
class Instrumentor:
    """埋点器子智能体：决定埋点位置、编辑 C 源、触发重编译烧录（需求 6）。

    依赖经构造注入，遵循平台的 Factory + Base-class 约定：

    * ``llm``：:class:`BaseLLMProvider`，供决定埋点位置等**推理**步骤使用；确定性的
      源码编辑不依赖它，以保持可离线复现。
    * ``device``：:class:`DeviceIO`，承载重编译/烧录等**副作用**（需求 6.3）。
    * ``tick_resolution_us``：目标平台默认时间戳/ tick 的最小分辨率（微秒），用于
      精度守卫（需求 6.6）。
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        device: DeviceIO,
        *,
        tick_resolution_us: int = SUBMS_RESOLUTION_US,
        required_resolution_us: int = SUBMS_RESOLUTION_US,
    ) -> None:
        self._llm = llm
        self._device = device
        self._tick_resolution_us = tick_resolution_us
        self._required_resolution_us = required_resolution_us

    @property
    def llm(self) -> BaseLLMProvider:
        """注入的 LLM 提供方（供推理步骤使用）。"""
        return self._llm

    @property
    def device(self) -> DeviceIO:
        """注入的设备控制面（承载重编译/烧录副作用）。"""
        return self._device

    def report_precision(self) -> PrecisionReport:
        """按注入的 tick 分辨率评估时间戳精度（需求 6.6）。"""
        return check_timing_precision(
            self._tick_resolution_us,
            required_resolution_us=self._required_resolution_us,
        )

    def edit_sources(
        self,
        sources: dict[str, str],
        points: Sequence[InstrumentationPoint],
    ) -> dict[str, str]:
        """对给定源码集合按点位插入成对埋点，返回被修改文件的新源码（纯编辑）。

        ``sources`` 为 ``file -> 源码`` 映射；每个 ``point.file`` 必须存在于其中。
        仅返回发生改动的文件，未涉及的文件不回传。
        """
        by_file: dict[str, list[InstrumentationPoint]] = {}
        for point in points:
            if point.file not in sources:
                raise KeyError(f"埋点引用了未提供的源码文件：{point.file!r}")
            by_file.setdefault(point.file, []).append(point)

        edited: dict[str, str] = {}
        for file, file_points in by_file.items():
            edited[file] = insert_many(sources[file], file_points)
        return edited

    def instrument(
        self,
        sources: dict[str, str],
        points: Sequence[InstrumentationPoint],
        *,
        write: Callable[[str, str], None] | None = None,
        reflash: bool = True,
    ) -> InstrumentationResult:
        """执行一次下钻埋点：编辑源码 →（可选落盘）→ 重编译烧录 → 汇总结果。

        Args:
            sources: ``file -> 源码`` 映射，提供待埋点文件的当前内容。
            points: 待插入的成对埋点点位（需求 6.1）。
            write: 可选的落盘回调 ``(file, new_source) -> None``；注入以隔离文件副作用，
                缺省则不落盘（仅返回编辑后的源码，便于测试/预览）。
            reflash: 是否在埋点后经 DeviceIO 触发重编译与烧录（需求 6.3）。

        Returns:
            :class:`InstrumentationResult`：含编辑后源码、点位/标签、精度报告，以及
            （若 ``reflash``）构建与烧录结果。
        """
        edited = self.edit_sources(sources, points)

        if write is not None:
            for file, new_source in edited.items():
                write(file, new_source)

        # 标签去重且保序，供 Orchestrator 记录到 instrumentation_history。
        tags: list[str] = []
        for point in points:
            if point.tag not in tags:
                tags.append(point.tag)

        precision = self.report_precision()

        build: BuildResult | InterventionRequest | None = None
        flash: FlashResult | InterventionRequest | None = None
        reflashed = False
        if reflash:
            # 需求 6.3：新埋点插入后经 DeviceIO 重编译并烧录目标。
            build = self._device.build()
            if isinstance(build, BuildResult) and build.success:
                flash = self._device.flash()
                reflashed = isinstance(flash, FlashResult) and flash.success

        return InstrumentationResult(
            edited_sources=edited,
            points=list(points),
            tags=tags,
            precision=precision,
            build=build,
            flash=flash,
            reflashed=reflashed,
        )
