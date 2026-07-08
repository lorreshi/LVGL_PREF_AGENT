"""语言无关 API 边界的 JSON 可序列化 DTO（任务 17.1）。

对照 design.md「语言无关的 API 边界」与需求 15：核心能力经一层与实现语言解耦的 API
暴露，确定性工件（尤其 ``FrameReport`` 及其火焰图字段：慢帧 / 热点 / 调用树引用）
序列化为 JSON，为未来 TS 客户端 / Web UI 留口子（Property 9）。

这些 DTO 只承载请求/响应的边界形态；能力产出的工件（如 ``FrameReport`` /
``KnowledgeRecord``）直接复用 ``core.models`` 的 Pydantic 契约以 JSON 形态透传。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "CreateRunRequest",
    "CreateRunResponse",
    "RunStatusResponse",
    "ReportResponse",
    "KnowledgeListResponse",
    "ResumeRequest",
    "ErrorResponse",
]


class CreateRunRequest(BaseModel):
    """``POST /runs`` 请求体：发起一次性能调优运行。"""

    scenario: str = Field(..., description="待复现/调优的场景描述")
    mode: str | None = Field(None, description="自动化模式 full_auto / half_auto；缺省取配置值")


class CreateRunResponse(BaseModel):
    """``POST /runs`` 响应体：返回本次运行的 ``thread_id``（需求 15.1）。"""

    thread_id: str
    status: str


class RunStatusResponse(BaseModel):
    """``GET /runs/{id}`` 响应体：运行状态概览（需求 15.2）。"""

    thread_id: str
    scenario: str
    mode: str
    iteration: int
    done: bool
    has_report: bool
    interrupted: bool = False


class ReportResponse(BaseModel):
    """``GET /runs/{id}/report`` 响应体：含火焰图字段的帧报告（需求 15.3/15.4）。"""

    thread_id: str
    report: dict[str, Any] | None


class KnowledgeListResponse(BaseModel):
    """``GET /knowledge`` 响应体：知识记录清单（需求 15）。"""

    records: list[dict[str, Any]] = Field(default_factory=list)


class ResumeRequest(BaseModel):
    """``POST /runs/{id}/resume`` 请求体：为被 interrupt 暂停的运行提供恢复值。"""

    value: Any = Field(None, description="人在环恢复值（如补充的场景、批准结果）")


class ErrorResponse(BaseModel):
    """统一错误响应体：携带缺失标识符与原因（需求 15.6）。"""

    error: str
    identifier: str | None = None
