"""任务 6.2：底座配置装配链集成测试（Tier 2）+ Property 6（知识跨线程可见）。

本测试把「配置 → 工厂 → 底座」这条装配链端到端串起来验证，严格对照
design.md "平台层 → Core_Services" 章节与需求 9.5 / 18.1 / 18.4：

* **装配链**（需求 18.1 / 18.4）：从一份 YAML 出发，经 ``Config_Loader`` 校验后
  由 :func:`build_core_services` **一次性**构造出全套工厂实例
  （LLM / Device / Retriever）并组装进 :class:`CoreServices`，各字段
  （device / llm / retriever / checkpointer / store / config）均接线完毕。
  既覆盖仓库自带的 ``config/config.example.yaml``，也覆盖临时写入的最小 YAML。

* **Property 6：知识跨线程可见**（需求 9.5）：任一被持久化的
  ``KnowledgeRecord`` 可被后续任意 ``thread_id`` 的 ``recall`` 召回。
  Store 是**跨线程**的长期记忆面，与线程范围的 Checkpointer 相互独立；只要平台
  持有的同一 Store 被注入各 Retriever，一个"线程"``persist`` 的记录即对另一个
  "线程"``recall`` 可见——与 ``thread_id`` 无关。

装配过程只做**构造**、不触达真实硬件 / 网络 / LLM：SerialDeviceIO 采用惰性开
串口（构造时不打开），AnthropicProvider 仅在 ``get_chat_model`` 时才读密钥，
本测试均不触发；Retriever 走本地 ``InMemoryStore``。因此无需任何 API key /
串口设备即可运行。

**Validates: Requirements 9.5, 18.1, 18.4**
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from embedded_device_agent.core.config.models import AppConfig
from embedded_device_agent.core.device.base import DeviceIO
from embedded_device_agent.core.llm.base import BaseLLMProvider
from embedded_device_agent.core.memory.backends.local_store import (
    DEFAULT_CAPABILITY,
    LocalStoreRetriever,
)
from embedded_device_agent.core.memory.base import BaseRetriever
from embedded_device_agent.core.models import KnowledgeRecord
from embedded_device_agent.core.services import CoreServices, build_core_services


# 仓库自带示例配置（合法、可经 Config_Loader 校验）。
_EXAMPLE_YAML = (
    Path(__file__).resolve().parents[2] / "config" / "config.example.yaml"
)

# 一份最小但合法的 YAML。device 用 serial：SerialDeviceIO 惰性开串口（构造时不
# 打开），故装配阶段不触达任何真实硬件。
_MINIMAL_YAML = """
llm:
  type: anthropic
  model: claude-3-5-sonnet-latest
  api_key_env: ANTHROPIC_API_KEY
device:
  type: serial
  port: /dev/ttyUSB0
  baud: 921600
  max_safe_baud: 921600
retriever:
  type: local_store
  top_k: 5
frame_budget_us: 16667
mode: full_auto
max_iterations: 5
"""


# ---------------------------------------------------------------------------
# 装配链：从 YAML 构造出全套工厂实例并组装 CoreServices（需求 18.1 / 18.4）
# ---------------------------------------------------------------------------
def _assert_fully_wired(cs: CoreServices) -> None:
    """断言 CoreServices 六个字段均被正确接线成对应基础设施类型。"""
    assert isinstance(cs, CoreServices)
    assert isinstance(cs.device, DeviceIO)
    assert isinstance(cs.llm, BaseLLMProvider)
    assert isinstance(cs.retriever, BaseRetriever)
    assert isinstance(cs.checkpointer, BaseCheckpointSaver)
    assert isinstance(cs.store, BaseStore)
    assert isinstance(cs.config, AppConfig)


def test_build_core_services_from_example_yaml_wires_all() -> None:
    """从仓库示例 YAML 经 Config_Loader 一次性装配出接好线的底座。"""
    assert _EXAMPLE_YAML.exists(), f"示例配置缺失：{_EXAMPLE_YAML}"

    cs = build_core_services(_EXAMPLE_YAML)

    _assert_fully_wired(cs)
    # 配置内容随 YAML 落地（示例：serial 设备 + local_store 检索器）。
    assert cs.config.device.type == "serial"
    assert cs.config.retriever.type == "local_store"
    # 默认后端选中本地 Store 检索器，且与平台持有的同一 Store 共享实例（需求 9.5）。
    assert isinstance(cs.retriever, LocalStoreRetriever)
    assert cs.retriever._store is cs.store


def test_build_core_services_from_temp_yaml(tmp_path: Path) -> None:
    """从临时写入的最小 YAML 装配，验证装配链不依赖仓库固定文件。"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_MINIMAL_YAML, encoding="utf-8")

    cs = build_core_services(cfg_path)

    _assert_fully_wired(cs)
    assert cs.config.device.type == "serial"
    assert cs.config.max_iterations == 5


def test_build_core_services_injects_shared_store_and_checkpointer() -> None:
    """显式注入的 Checkpointer / Store 被原样采用，且本地 Retriever 复用该 Store。"""
    store = InMemoryStore()
    cs = build_core_services(_example_appconfig(), store=store)

    assert cs.store is store
    assert isinstance(cs.retriever, LocalStoreRetriever)
    # 平台持有的 Store 与 Retriever 内部 Store 是同一实例——跨线程可见的前提。
    assert cs.retriever._store is store


def _example_appconfig() -> AppConfig:
    """由最小 YAML 构造 AppConfig，供注入路径测试直接复用（避免落盘临时文件）。"""
    from embedded_device_agent.core.config.loader import ConfigLoader

    return ConfigLoader.from_string(_MINIMAL_YAML)


# ---------------------------------------------------------------------------
# 记忆链 + Property 6：知识跨线程可见（需求 9.5）
# ---------------------------------------------------------------------------
def _make_record(symptom: str, *, created_at: datetime, id_: str | None = None) -> KnowledgeRecord:
    return KnowledgeRecord(
        id=id_,
        symptom=symptom,
        root_cause="过度重绘导致主线程阻塞",
        optimization="启用局部刷新 + 缓存图层",
        effect="baseline 30fps -> 60fps",
        scenario="scroll",
        created_at=created_at,
    )


def test_memory_chain_persist_then_recall_cross_thread() -> None:
    """记忆链示例：一个'线程' persist 的记录可被持有同一 Store 的另一'线程' recall。

    以两套共享同一注入 Store 的 CoreServices 模拟不同 ``thread_id`` 的运行：
    Checkpointer（线程范围短期状态）各自独立，Store（跨线程长期记忆）为同一实例。
    """
    store = InMemoryStore()
    thread_a = build_core_services(_example_appconfig(), store=store)
    thread_b = build_core_services(_example_appconfig(), store=store)

    # 两个"线程"的短期状态相互独立，长期记忆面则共享同一 Store。
    assert thread_a.checkpointer is not thread_b.checkpointer
    assert thread_a.store is thread_b.store

    rec_id = thread_a.retriever.persist(
        _make_record("滚动列表卡顿掉帧", created_at=datetime(2024, 1, 1, tzinfo=timezone.utc), id_="rec-x")
    )
    recalled = thread_b.retriever.recall("滚动 卡顿", k=5)

    assert [r.id for r in recalled] == [rec_id]
    assert recalled[0].symptom == "滚动列表卡顿掉帧"


# 症状生成器：1~4 个 ASCII token（含数字），保证分词后必有可匹配 token。
_symptom_token = st.from_regex(r"[a-z][a-z0-9]{0,11}", fullmatch=True)
_symptoms = st.lists(_symptom_token, min_size=1, max_size=4).map(" ".join)


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    symptom=_symptoms,
    created_at=st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2100, 1, 1),
        timezones=st.just(timezone.utc),
    ),
    thread_count=st.integers(min_value=1, max_value=4),
)
def test_property6_knowledge_visible_across_threads(
    symptom: str, created_at: datetime, thread_count: int
) -> None:
    """Property 6：任一被持久化的 KnowledgeRecord 可被后续任意 thread_id 的 recall 召回。

    **Validates: Requirements 9.5**

    经 :func:`build_core_services` 装配的底座共享同一注入 Store。一个"线程"
    persist 记录后，其余任意多个新建"线程"（各自独立 Checkpointer）都应能
    ``recall`` 到该记录——可见性与 ``thread_id`` 无关。
    """
    store = InMemoryStore()

    # 写入线程：装配一套底座并持久化一条知识。
    writer = build_core_services(_example_appconfig(), store=store)
    rec_id = writer.retriever.persist(_make_record(symptom, created_at=created_at))

    # 后续任意多个"线程"：各自独立装配（独立 Checkpointer），共享同一 Store。
    for _ in range(thread_count):
        reader = build_core_services(_example_appconfig(), store=store)
        # 每个 reader 的短期状态独立，长期记忆面共享。
        assert reader.checkpointer is not writer.checkpointer
        assert reader.store is store

        recalled = reader.retriever.recall(symptom, k=10)
        recalled_ids = {r.id for r in recalled}
        assert rec_id in recalled_ids, (
            f"跨线程召回失败：persist 的记录 {rec_id!r} 未被新线程 recall 到"
            f"（symptom={symptom!r}）"
        )
