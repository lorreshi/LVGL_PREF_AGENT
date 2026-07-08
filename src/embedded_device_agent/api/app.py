"""语言无关 API 边界：FastAPI 应用（任务 17.1）。

对照 design.md「语言无关的 API 边界」与需求 15：把 Capability A 的调优闭环经一层与
实现语言解耦的 HTTP/JSON API 暴露。路由：

* ``POST /runs``——发起一次调优运行，返回 ``thread_id``（需求 15.1）。
* ``GET /runs/{id}``——查询运行状态（需求 15.2）。
* ``GET /runs/{id}/report``——返回含火焰图字段的 ``FrameReport`` JSON（需求 15.3/15.4）。
* ``GET /knowledge``——返回沉淀的知识记录（可选 ``symptom`` 语义召回）（需求 15.5）。
* ``POST /runs/{id}/resume``——为 Half_Auto 干预点被 interrupt 暂停的运行提供恢复值。

引用不存在的 run/record 时，返回带缺失标识符的错误（需求 15.6）。

**安全提示**：本 API 默认**无鉴权**（需求未要求，但生产对外暴露时必须另加访问控制，
如反向代理鉴权 / API Key / mTLS）。请勿在未加保护的情况下将其暴露到不受信网络。
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from langgraph.types import Command

from embedded_device_agent.api.dto import (
    CreateRunRequest,
    CreateRunResponse,
    KnowledgeListResponse,
    ReportResponse,
    ResumeRequest,
    RunStatusResponse,
)
from embedded_device_agent.capabilities.perf_tuning.graph import build_tuning_graph
from embedded_device_agent.core.memory.backends.local_store import (
    DEFAULT_CAPABILITY,
    NAMESPACE_ROOT,
)
from embedded_device_agent.core.services import CoreServices

__all__ = ["create_app"]

# 每次运行的 recursion_limit，与能力工具保持一致，容纳多轮闭环。
_RUN_RECURSION_LIMIT = 500


def create_app(core: CoreServices) -> FastAPI:
    """用注入的共享底座构建 FastAPI 应用（需求 15 / 18.4）。

    调优闭环图以 ``core.checkpointer`` 编译一次并复用；各运行以 ``thread_id`` 区分，
    状态由 Checkpointer 持久化，故 ``GET`` 系列端点从检查点读取，``resume`` 从断点恢复。
    """
    app = FastAPI(title="Embedded Device Agent API", version="0.1.0")
    graph = build_tuning_graph(core)

    def _thread_config(thread_id: str) -> dict[str, Any]:
        return {"recursion_limit": _RUN_RECURSION_LIMIT, "configurable": {"thread_id": thread_id}}

    def _get_state(thread_id: str):
        """读取某运行的检查点状态；无检查点则视为不存在（需求 15.6）。"""
        snapshot = graph.get_state(_thread_config(thread_id))
        if not snapshot.values:
            raise HTTPException(
                status_code=404,
                detail={"error": "run not found", "identifier": thread_id},
            )
        return snapshot

    @app.post("/runs", response_model=CreateRunResponse)
    def create_run(req: CreateRunRequest) -> CreateRunResponse:
        """发起一次调优运行，返回 thread_id（需求 15.1）。"""
        thread_id = f"run-{uuid.uuid4().hex}"
        result = graph.invoke(
            {"scenario": req.scenario, "mode": req.mode or core.config.mode},
            config=_thread_config(thread_id),
        )
        # 运行可能已完成，或在 Half_Auto 干预点被 interrupt 暂停。
        interrupted = "__interrupt__" in result
        status = "interrupted" if interrupted else ("done" if result.get("done") else "running")
        return CreateRunResponse(thread_id=thread_id, status=status)

    @app.get("/runs/{thread_id}", response_model=RunStatusResponse)
    def get_run(thread_id: str) -> RunStatusResponse:
        """查询运行状态（需求 15.2）；不存在返回带 thread_id 的错误（需求 15.6）。"""
        snapshot = _get_state(thread_id)
        values = snapshot.values
        return RunStatusResponse(
            thread_id=thread_id,
            scenario=values.get("scenario", ""),
            mode=values.get("mode", core.config.mode),
            iteration=values.get("iteration", 0),
            done=bool(values.get("done")),
            has_report=values.get("latest") is not None,
            interrupted=bool(snapshot.next),
        )

    @app.get("/runs/{thread_id}/report", response_model=ReportResponse)
    def get_report(thread_id: str) -> ReportResponse:
        """返回含火焰图字段的 FrameReport JSON（需求 15.3/15.4）。"""
        values = _get_state(thread_id).values
        report = values.get("latest")
        return ReportResponse(
            thread_id=thread_id,
            report=report.model_dump(mode="json") if report is not None else None,
        )

    @app.post("/runs/{thread_id}/resume", response_model=RunStatusResponse)
    def resume_run(thread_id: str, req: ResumeRequest) -> RunStatusResponse:
        """为被 interrupt 暂停的运行提供恢复值并继续（需求 10.2/10.3）。"""
        _get_state(thread_id)  # 校验存在性（不存在则 404）。
        graph.invoke(Command(resume=req.value), config=_thread_config(thread_id))
        return get_run(thread_id)

    @app.get("/knowledge", response_model=KnowledgeListResponse)
    def list_knowledge(symptom: str | None = None, k: int = 20) -> KnowledgeListResponse:
        """返回沉淀的知识记录（需求 15.5）。

        提供 ``symptom`` 时按其语义召回；否则枚举 perf_tuning 分区下的全部记录。
        """
        if symptom:
            records = core.retriever.recall(symptom, k)
            return KnowledgeListResponse(records=[r.model_dump(mode="json") for r in records])
        # 无症状：直接从共享 Store 枚举本能力分区下的记录（语言中立 JSON）。
        items = core.store.search((NAMESPACE_ROOT, DEFAULT_CAPABILITY), limit=k)
        return KnowledgeListResponse(records=[dict(item.value) for item in items])

    return app
