"""任务 5.2：LocalStoreRetriever / ExternalRAGRetriever 后端单元测试。

覆盖 design.md "Retriever" 章节与需求 9.2 / 9.4 / 18.3 的关键性质：

* **工厂 ``type`` 分发**：``RetrieverFactory.create`` 依 ``cfg.type`` 构造
  ``local_store`` → ``LocalStoreRetriever``、``external_rag`` →
  ``ExternalRAGRetriever``（需求 9.4 / 12.5）；
* **召回相关性排序**（需求 9.2）：与症状更相关的记录排在前；``k`` 限制生效；
  无相关知识时返回空列表；
* **跨能力分区**（需求 18.3）：不同能力命名空间下的记录相互隔离；
* **跨线程可见**（需求 9.5）：经共享注入 Store，任一"线程"persist 的记录可被
  另一"线程"recall 召回。

以注入的共享 ``InMemoryStore`` 保证确定性，不触达真实持久化后端。

_Requirements: 9.2, 9.4, 18.3_
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from embedded_device_agent.core.config.models import RetrieverConfig
from embedded_device_agent.core.memory import RetrieverFactory
from embedded_device_agent.core.memory.backends.local_store import (
    DEFAULT_CAPABILITY,
    ExternalRAGRetriever,
    LocalStoreRetriever,
)
from embedded_device_agent.core.models import KnowledgeRecord

from langgraph.store.memory import InMemoryStore


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------
_BASE_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_cfg(type_: str = "local_store", top_k: int = 5) -> RetrieverConfig:
    return RetrieverConfig(type=type_, top_k=top_k)


def _make_record(
    symptom: str,
    *,
    root_cause: str = "rc",
    optimization: str = "opt",
    effect: str = "baseline 30fps -> 60fps",
    scenario: str = "scroll",
    created_at: datetime | None = None,
    id_: str | None = None,
) -> KnowledgeRecord:
    return KnowledgeRecord(
        id=id_,
        symptom=symptom,
        root_cause=root_cause,
        optimization=optimization,
        effect=effect,
        scenario=scenario,
        created_at=created_at or _BASE_TS,
    )


def _local(store: InMemoryStore, capability: str = DEFAULT_CAPABILITY) -> LocalStoreRetriever:
    """构造一个注入共享 store 与能力分区的 LocalStoreRetriever。"""
    return LocalStoreRetriever(_make_cfg(), store=store, capability=capability)


# ---------------------------------------------------------------------------
# 工厂：依 type 分发构造具体后端（需求 9.4 / 12.5）
# ---------------------------------------------------------------------------
def test_factory_dispatches_local_store():
    retriever = RetrieverFactory.create(_make_cfg("local_store"))
    assert isinstance(retriever, LocalStoreRetriever)


def test_factory_dispatches_external_rag():
    cfg = RetrieverConfig(type="external_rag", endpoint="http://rag", collection="kb")
    retriever = RetrieverFactory.create(cfg)
    assert isinstance(retriever, ExternalRAGRetriever)
    assert retriever.endpoint == "http://rag"
    assert retriever.collection == "kb"


def test_external_rag_recall_persist_raise_until_wired():
    """外部 RAG 骨架未接线前应主动报错，避免静默返回空知识。"""
    retriever = ExternalRAGRetriever(RetrieverConfig(type="external_rag"))
    with pytest.raises(NotImplementedError):
        retriever.recall("卡顿", k=3)
    with pytest.raises(NotImplementedError):
        retriever.persist(_make_record("卡顿"))


# ---------------------------------------------------------------------------
# 持久化 + 召回：基础往返
# ---------------------------------------------------------------------------
def test_persist_returns_id_and_recall_finds_it():
    store = InMemoryStore()
    retriever = _local(store)
    rec_id = retriever.persist(_make_record("滚动列表卡顿掉帧"))
    assert isinstance(rec_id, str) and rec_id

    recalled = retriever.recall("滚动 卡顿", k=5)
    assert len(recalled) == 1
    assert recalled[0].symptom == "滚动列表卡顿掉帧"
    assert recalled[0].id == rec_id


def test_persist_preserves_explicit_id():
    store = InMemoryStore()
    retriever = _local(store)
    rec_id = retriever.persist(_make_record("卡顿", id_="fixed-id"))
    assert rec_id == "fixed-id"
    assert retriever.recall("卡顿", k=5)[0].id == "fixed-id"


# ---------------------------------------------------------------------------
# 召回相关性排序（需求 9.2）
# ---------------------------------------------------------------------------
def test_recall_orders_more_relevant_first():
    """症状字段命中更多的记录应排在前。"""
    store = InMemoryStore()
    retriever = _local(store)
    # 高相关：症状直接命中 "滚动 卡顿"
    retriever.persist(_make_record("滚动 卡顿 严重", id_="high"))
    # 低相关：仅在其它字段沾边，症状不含查询词
    retriever.persist(
        _make_record(
            "屏幕闪烁",
            root_cause="滚动时偶发",
            scenario="idle",
            id_="low",
        )
    )

    recalled = retriever.recall("滚动 卡顿", k=5)
    ids = [r.id for r in recalled]
    assert ids[0] == "high"
    assert ids.index("high") < ids.index("low")


def test_recall_respects_k_limit():
    store = InMemoryStore()
    retriever = _local(store)
    for i in range(5):
        retriever.persist(_make_record("滚动卡顿掉帧", id_=f"rec-{i}"))
    recalled = retriever.recall("滚动卡顿", k=2)
    assert len(recalled) == 2


def test_recall_empty_when_no_match():
    store = InMemoryStore()
    retriever = _local(store)
    retriever.persist(_make_record("滚动列表卡顿"))
    assert retriever.recall("完全无关的天气预报", k=5) == []


def test_recall_empty_for_nonpositive_k():
    store = InMemoryStore()
    retriever = _local(store)
    retriever.persist(_make_record("滚动卡顿"))
    assert retriever.recall("滚动", k=0) == []
    assert retriever.recall("滚动", k=-1) == []


def test_recall_empty_for_blank_symptom():
    store = InMemoryStore()
    retriever = _local(store)
    retriever.persist(_make_record("滚动卡顿"))
    assert retriever.recall("", k=5) == []


def test_recall_ties_break_by_recency():
    """相关性相同时，更新时间更近的记录优先。"""
    store = InMemoryStore()
    retriever = _local(store)
    retriever.persist(
        _make_record("滚动卡顿", created_at=_BASE_TS, id_="older")
    )
    retriever.persist(
        _make_record(
            "滚动卡顿", created_at=_BASE_TS + timedelta(hours=1), id_="newer"
        )
    )
    recalled = retriever.recall("滚动卡顿", k=5)
    assert [r.id for r in recalled] == ["newer", "older"]


# ---------------------------------------------------------------------------
# 跨能力分区（需求 18.3）
# ---------------------------------------------------------------------------
def test_cross_capability_partitions_are_isolated():
    """不同能力命名空间下的记录相互隔离，互不召回。"""
    store = InMemoryStore()
    perf = _local(store, capability="perf_tuning")
    power = _local(store, capability="power_opt")

    perf.persist(_make_record("滚动卡顿", id_="perf-rec"))
    power.persist(_make_record("滚动卡顿", id_="power-rec"))

    perf_recalled = perf.recall("滚动卡顿", k=5)
    power_recalled = power.recall("滚动卡顿", k=5)

    assert [r.id for r in perf_recalled] == ["perf-rec"]
    assert [r.id for r in power_recalled] == ["power-rec"]


def test_recall_empty_in_untouched_capability():
    store = InMemoryStore()
    perf = _local(store, capability="perf_tuning")
    other = _local(store, capability="new_capability")

    perf.persist(_make_record("滚动卡顿"))
    assert other.recall("滚动卡顿", k=5) == []


# ---------------------------------------------------------------------------
# 跨线程可见（需求 9.5）：共享 Store 不绑定线程
# ---------------------------------------------------------------------------
def test_cross_thread_visibility_via_shared_store():
    """一个线程 persist 的记录可被持有同一 Store 的另一线程召回。"""
    shared = InMemoryStore()
    thread_a = _local(shared)
    thread_b = _local(shared)

    rec_id = thread_a.persist(_make_record("滚动列表卡顿", id_="shared-rec"))
    recalled = thread_b.recall("滚动 卡顿", k=5)

    assert [r.id for r in recalled] == [rec_id]
