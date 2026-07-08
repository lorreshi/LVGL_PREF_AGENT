"""任务 17.2：语言无关 API 边界集成测试（Tier 2）。

对照需求 15.3/15.4/15.5/15.6 与 **Property 9：工件语言中立**——``POST /runs`` 返回
thread_id；``GET /runs/{id}/report`` 返回含火焰图字段的 JSON；引用不存在的标识符报错
（带缺失标识符）。

以 ``fastapi.testclient.TestClient`` 驱动，用 harness fake 构造 CoreServices，不触达真实
硬件 / LLM / 网络。

**Validates: Requirements 15.3, 15.4, 15.5, 15.6**
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from embedded_device_agent.api.app import create_app
from embedded_device_agent.core.config.models import (
    AppConfig,
    DeviceConfig,
    LLMConfig,
    RetrieverConfig,
)
from embedded_device_agent.core.memory.backends.local_store import LocalStoreRetriever
from embedded_device_agent.core.services import CoreServices
from tests.harness import FakeDeviceIO, FakeLLMProvider


@pytest.fixture
def client() -> TestClient:
    store = InMemoryStore()
    cfg = AppConfig(
        llm=LLMConfig(type="anthropic", model="m", api_key_env="ANTHROPIC_API_KEY"),
        device=DeviceConfig(type="fake", port="/dev/ttyFAKE0"),
        retriever=RetrieverConfig(),
        frame_budget_us=1_000_000,  # 无慢帧 → 一轮达标并沉淀知识
        mode="full_auto",
        max_iterations=3,
    )
    core = CoreServices(
        device=FakeDeviceIO(),
        llm=FakeLLMProvider(responses=["ok"]),
        retriever=LocalStoreRetriever(cfg.retriever, store=store),
        checkpointer=InMemorySaver(),
        store=store,
        config=cfg,
    )
    return TestClient(create_app(core))


# --------------------------------------------------------------------------- #
# 需求 15.1/15.2：POST /runs 返回 thread_id；GET /runs/{id} 查询状态
# --------------------------------------------------------------------------- #
def test_post_runs_returns_thread_id(client: TestClient) -> None:
    resp = client.post("/runs", json={"scenario": "滚动列表卡顿"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"]
    assert body["status"] in {"done", "running", "interrupted"}


def test_get_run_status(client: TestClient) -> None:
    thread_id = client.post("/runs", json={"scenario": "滚动"}).json()["thread_id"]
    resp = client.get(f"/runs/{thread_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"] == thread_id
    assert body["scenario"] == "滚动"
    assert body["done"] is True


# --------------------------------------------------------------------------- #
# 需求 15.3/15.4 + Property 9：report 返回含火焰图字段的 JSON
# --------------------------------------------------------------------------- #
def test_get_report_returns_flame_graph_fields_json(client: TestClient) -> None:
    thread_id = client.post("/runs", json={"scenario": "滚动"}).json()["thread_id"]
    resp = client.get(f"/runs/{thread_id}/report")
    assert resp.status_code == 200
    report = resp.json()["report"]
    assert report is not None
    # 火焰图字段：热点 / 慢帧 / 调用树引用 + 聚合统计（语言中立 JSON，Property 9）。
    for field in (
        "hotspots",
        "slow_frames",
        "source",
        "total_frames",
        "p95_frame_us",
        "frame_budget_us",
    ):
        assert field in report, f"report 缺少火焰图/统计字段 {field}"


# --------------------------------------------------------------------------- #
# 需求 15.5：GET /knowledge 返回知识记录
# --------------------------------------------------------------------------- #
def test_get_knowledge_lists_records(client: TestClient) -> None:
    # 先跑一次运行以沉淀一条知识（无慢帧达标即持久化）。
    client.post("/runs", json={"scenario": "滚动列表卡顿"})
    resp = client.get("/knowledge")
    assert resp.status_code == 200
    records = resp.json()["records"]
    assert isinstance(records, list)
    assert len(records) >= 1
    # 记录为语言中立 JSON，含关键字段。
    assert "symptom" in records[0]


# --------------------------------------------------------------------------- #
# 需求 15.6：引用不存在的 run 返回带缺失标识符的错误
# --------------------------------------------------------------------------- #
def test_missing_run_returns_error_with_identifier(client: TestClient) -> None:
    resp = client.get("/runs/does-not-exist")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["identifier"] == "does-not-exist"
    assert "not found" in detail["error"]


def test_missing_run_report_returns_error_with_identifier(client: TestClient) -> None:
    resp = client.get("/runs/nope-123/report")
    assert resp.status_code == 404
    assert resp.json()["detail"]["identifier"] == "nope-123"
