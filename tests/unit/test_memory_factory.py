"""任务 5.1：Memory / Retriever 底座（BaseRetriever + RetrieverFactory）单元测试。

验证工厂机制与抽象契约，不触达真实 Store / 外部 RAG：
* 装饰器注册使子类进入注册表；
* ``create(cfg)`` 依 ``cfg.type`` 分发构造正确子类，并把 cfg 透传给构造函数；
* 未知 ``type`` 报出清晰错误（列出可用类型）；
* 非 BaseRetriever 子类注册被拒绝、重复注册同一 key 被拒绝；
* ``BaseRetriever`` 抽象不可直接实例化；
* 用一个 DummyRetriever 验证 ``recall`` / ``persist`` 接口约定（内存实现，无真实存储）。

_Requirements: 9.4, 12.5_
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from embedded_device_agent.core.config.models import RetrieverConfig
from embedded_device_agent.core.memory import BaseRetriever, RetrieverFactory
from embedded_device_agent.core.models import KnowledgeRecord


@pytest.fixture
def isolated_registry():
    """隔离工厂注册表：每个用例前后清空并还原，避免用例间串扰。"""
    saved = dict(RetrieverFactory._registry)
    RetrieverFactory._registry.clear()
    try:
        yield RetrieverFactory
    finally:
        RetrieverFactory._registry.clear()
        RetrieverFactory._registry.update(saved)


def _make_cfg(type_: str = "local_store", top_k: int = 5) -> RetrieverConfig:
    return RetrieverConfig(type=type_, top_k=top_k)


def _make_record(symptom: str) -> KnowledgeRecord:
    return KnowledgeRecord(
        symptom=symptom,
        root_cause="rc",
        optimization="opt",
        effect="baseline 30fps -> 60fps",
        scenario="scroll",
        created_at=datetime.now(timezone.utc),
    )


class _DummyRetriever(BaseRetriever):
    """最小内存 Retriever，用于测工厂机制与接口约定（不触达真实存储）。"""

    def __init__(self, cfg: RetrieverConfig) -> None:
        self.cfg = cfg
        self._store: list[KnowledgeRecord] = []

    def recall(self, symptom: str, k: int) -> list[KnowledgeRecord]:
        # 朴素子串匹配 + 截断至 k，仅用于验证接口约定
        matches = [r for r in self._store if symptom in r.symptom]
        return matches[:k]

    def persist(self, record: KnowledgeRecord) -> str:
        rec_id = f"rec-{len(self._store)}"
        stored = record.model_copy(update={"id": rec_id})
        self._store.append(stored)
        return rec_id


# ---------------------------------------------------------------------------
# 工厂：注册与按 type 分发
# ---------------------------------------------------------------------------
def test_register_decorator_adds_to_registry(isolated_registry):
    @isolated_registry.register("dummy")
    class Dummy(_DummyRetriever):
        pass

    assert isolated_registry._registry["dummy"] is Dummy


def test_create_dispatches_by_type_and_passes_cfg(isolated_registry):
    @isolated_registry.register("dummy")
    class Dummy(_DummyRetriever):
        pass

    cfg = _make_cfg("dummy")
    retriever = isolated_registry.create(cfg)

    assert isinstance(retriever, Dummy)
    assert isinstance(retriever, BaseRetriever)
    assert retriever.cfg is cfg


def test_create_dispatches_among_multiple_types(isolated_registry):
    @isolated_registry.register("alpha")
    class Alpha(_DummyRetriever):
        pass

    @isolated_registry.register("beta")
    class Beta(_DummyRetriever):
        pass

    assert isinstance(isolated_registry.create(_make_cfg("alpha")), Alpha)
    assert isinstance(isolated_registry.create(_make_cfg("beta")), Beta)


def test_create_unknown_type_raises_clear_error(isolated_registry):
    @isolated_registry.register("known")
    class Known(_DummyRetriever):
        pass

    with pytest.raises(ValueError) as exc:
        isolated_registry.create(_make_cfg("does_not_exist"))

    msg = str(exc.value)
    assert "does_not_exist" in msg
    # 错误信息应列出可用类型，便于定位配置问题
    assert "known" in msg


def test_register_rejects_non_retriever_subclass(isolated_registry):
    with pytest.raises(TypeError):

        @isolated_registry.register("bad")
        class NotARetriever:  # 非 BaseRetriever 子类
            pass


def test_register_rejects_duplicate_key(isolated_registry):
    @isolated_registry.register("dup")
    class First(_DummyRetriever):
        pass

    with pytest.raises(ValueError):

        @isolated_registry.register("dup")
        class Second(_DummyRetriever):
            pass


def test_baseretriever_is_abstract():
    """BaseRetriever 不可直接实例化（抽象方法未实现）。"""
    with pytest.raises(TypeError):
        BaseRetriever()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 接口约定：recall / persist（用 dummy 内存实现验证）
# ---------------------------------------------------------------------------
def test_persist_returns_id_and_recall_finds_it(isolated_registry):
    retriever = _DummyRetriever(_make_cfg())
    rec_id = retriever.persist(_make_record("卡顿掉帧"))
    assert isinstance(rec_id, str) and rec_id

    recalled = retriever.recall("卡顿", k=5)
    assert len(recalled) == 1
    assert recalled[0].symptom == "卡顿掉帧"


def test_recall_empty_when_no_match(isolated_registry):
    retriever = _DummyRetriever(_make_cfg())
    retriever.persist(_make_record("卡顿掉帧"))
    assert retriever.recall("完全无关的症状", k=5) == []


def test_recall_respects_k_limit(isolated_registry):
    retriever = _DummyRetriever(_make_cfg())
    for _ in range(5):
        retriever.persist(_make_record("滚动卡顿"))
    assert len(retriever.recall("滚动", k=2)) == 2
