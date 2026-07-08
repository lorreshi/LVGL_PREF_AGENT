"""具体 Retriever 后端：LocalStoreRetriever 与 ExternalRAGRetriever（任务 5.2）。

严格对照 design.md "Retriever" 章节与需求 9.1 / 9.2 / 9.5 / 18.3：

* ``LocalStoreRetriever``——基于 **LangGraph Store**（本地起步，默认后端）的语义
  召回与持久化实现。所有长期记忆副作用都藏在本类之后，经
  ``@RetrieverFactory.register("local_store")`` 注册，由 YAML 的
  ``retriever.type == "local_store"`` 选择（需求 12.5）。关键性质：

  - **按能力标识分区**（需求 18.3）：``Knowledge_Record`` 以
    ``(namespace_root, capability)`` 作为 Store 命名空间存放，使跨能力的知识
    既可隔离又可按需共享。
  - **跨线程可见**（需求 9.5 / Property 6）：Store 是**跨线程**的长期记忆面，
    与线程范围的 Checkpointer 相互独立；任一线程 ``persist`` 的记录都可被后续
    任意 ``thread_id`` 的 ``recall`` 召回——因为本类持有的 Store 句柄不绑定任何
    线程。
  - **语义召回并按相关性排序**（需求 9.2）：``recall`` 依症状对分区内记录做
    相关性打分并降序返回至多 ``k`` 条；无相关知识时返回空列表。

* ``ExternalRAGRetriever``——以 **tool 形式调用外部 RAG** 的骨架（需求 9.4）。
  仅声明契约与注入点（``@RetrieverFactory.register("external_rag")``），具体
  HTTP/RAG 调用留待接入真实外部服务时实现；未接线前调用即报出清晰的未实现原因，
  避免静默返回空结果被误读为"无知识"。

依赖注入友好：``LocalStoreRetriever`` 默认自建一个 ``InMemoryStore``，测试与
``Core_Services`` 装配时可注入共享 Store（例如同一 ``InMemoryStore`` 实例以验证
跨线程可见），从而无需真实持久化后端即可对全部路径做单元测试。
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from embedded_device_agent.core.config.models import RetrieverConfig
from embedded_device_agent.core.memory.base import BaseRetriever
from embedded_device_agent.core.memory.factory import RetrieverFactory
from embedded_device_agent.core.models import KnowledgeRecord

__all__ = [
    "LocalStoreRetriever",
    "ExternalRAGRetriever",
    "DEFAULT_CAPABILITY",
    "NAMESPACE_ROOT",
]

# 命名空间根前缀：所有长期知识都挂在该前缀下，再按能力标识分区（需求 18.3）。
NAMESPACE_ROOT = "knowledge"
# 未显式指定能力标识时的默认分区（Capability A：性能调优）。
DEFAULT_CAPABILITY = "perf_tuning"

# 症状文本相关性打分中，症状字段命中相较其余字段命中的权重。
_SYMPTOM_WEIGHT = 2.0
_OTHER_WEIGHT = 1.0

# 分词：抽取 ASCII 单词（含数字）与单个 CJK 字符；CJK 再补充相邻二元组以提升区分度。
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokenize(text: str) -> set[str]:
    """把文本切成用于相关性打分的 token 集合。

    同时支持中英混排：
    * ASCII：按 ``[a-z0-9]+`` 抽取小写单词；
    * 中文：抽取单字，并追加相邻二元组（bigram）以提升短语区分度。
    """
    if not text:
        return set()
    lowered = text.lower()
    tokens: set[str] = set(_LATIN_TOKEN_RE.findall(lowered))
    cjk_chars = _CJK_CHAR_RE.findall(lowered)
    tokens.update(cjk_chars)
    tokens.update(
        cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)
    )
    return tokens


@RetrieverFactory.register("local_store")
class LocalStoreRetriever(BaseRetriever):
    """基于 LangGraph Store 的本地语义召回 / 持久化实现（默认后端）。

    工厂经 ``LocalStoreRetriever(cfg)`` 构造（见 ``RetrieverFactory.create``），故除
    ``cfg`` 外的参数均为可选：可注入共享 ``store``（跨线程/跨组件复用同一份长期
    记忆）与 ``capability``（能力标识分区）。
    """

    def __init__(
        self,
        cfg: RetrieverConfig,
        *,
        store: BaseStore | None = None,
        capability: str = DEFAULT_CAPABILITY,
    ) -> None:
        self.cfg = cfg
        # Store 是跨线程的长期记忆面：默认自建 InMemoryStore；装配/测试时可注入共享实例。
        self._store: BaseStore = store if store is not None else InMemoryStore()
        self._capability = capability
        # 单次枚举分区记录的上限，足够覆盖长期知识规模且避免无界扫描。
        self._scan_limit = 1000

    # -- 命名空间（按能力标识分区，需求 18.3）-------------------------------
    def _namespace(self, capability: str | None = None) -> tuple[str, str]:
        return (NAMESPACE_ROOT, capability or self._capability)

    # -- 持久化 -------------------------------------------------------------
    def persist(self, record: KnowledgeRecord) -> str:
        """持久化一条 ``KnowledgeRecord`` 到本能力分区并返回其存储 id（需求 9.1）。

        记录以 JSON 兼容 dict 存入 Store；缺 ``id`` 时生成一个稳定的 hex id。
        存入后即跨线程可见（需求 9.5）——Store 不绑定任何 ``thread_id``。
        """
        record_id = record.id or uuid.uuid4().hex
        stored = record.model_copy(update={"id": record_id})
        self._store.put(
            self._namespace(),
            record_id,
            stored.model_dump(mode="json"),
        )
        return record_id

    # -- 语义召回 -----------------------------------------------------------
    def recall(self, symptom: str, k: int) -> list[KnowledgeRecord]:
        """按 ``symptom`` 语义召回至多 ``k`` 条记录，按相关性降序排序（需求 9.2）。

        对本能力分区内所有记录做相关性打分（症状字段命中权重更高），仅返回有正向
        相关性的记录；无相关知识或 ``k <= 0`` 时返回空列表。
        """
        if k <= 0:
            return []
        query_tokens = _tokenize(symptom)
        if not query_tokens:
            return []

        items = self._store.search(self._namespace(), limit=self._scan_limit)

        scored: list[tuple[float, datetime, str, KnowledgeRecord]] = []
        for item in items:
            record = KnowledgeRecord.model_validate(item.value)
            score = self._relevance(query_tokens, record)
            if score > 0.0:
                scored.append((score, record.created_at, record.id or "", record))

        # 主序：相关性降序；次序：更新时间更近者优先；末序：id 稳定排序。
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        return [record for _, _, _, record in scored[:k]]

    # -- 相关性打分 ---------------------------------------------------------
    @staticmethod
    def _relevance(query_tokens: set[str], record: KnowledgeRecord) -> float:
        """症状 token 与记录文本的加权重叠计数：症状字段命中权重更高。"""
        symptom_tokens = _tokenize(record.symptom)
        other_tokens = _tokenize(
            " ".join(
                (
                    record.scenario,
                    record.root_cause,
                    record.optimization,
                    record.effect,
                )
            )
        )
        symptom_overlap = len(query_tokens & symptom_tokens)
        other_overlap = len(query_tokens & (other_tokens - symptom_tokens))
        return _SYMPTOM_WEIGHT * symptom_overlap + _OTHER_WEIGHT * other_overlap


@RetrieverFactory.register("external_rag")
class ExternalRAGRetriever(BaseRetriever):
    """以 tool 形式调用外部 RAG 的骨架（需求 9.4）。

    通过 YAML ``retriever.type == "external_rag"`` 选择即可用外部 RAG 后端替换本地
    Store 实现（需求 9.4 / 12.5）；``endpoint`` / ``collection`` 由 ``RetrieverConfig``
    提供。真实的外部检索/写入（HTTP/gRPC 或 RAG SDK 调用）留待接入具体服务时实现，
    此处仅固定注入点与契约。

    未接线前，``recall`` / ``persist`` 主动抛出清晰的未实现错误，避免静默返回空结果
    被上层误读为"无历史知识"而做出错误决策。
    """

    def __init__(self, cfg: RetrieverConfig) -> None:
        self.cfg = cfg
        self.endpoint = cfg.endpoint
        self.collection = cfg.collection

    def recall(self, symptom: str, k: int) -> list[KnowledgeRecord]:
        raise NotImplementedError(
            "ExternalRAGRetriever 为外部 RAG 接入骨架，recall 尚未接线。"
            "请在此实现对外部 RAG 服务的语义检索调用"
            f"（endpoint={self.endpoint!r}, collection={self.collection!r}）。"
        )

    def persist(self, record: KnowledgeRecord) -> str:
        raise NotImplementedError(
            "ExternalRAGRetriever 为外部 RAG 接入骨架，persist 尚未接线。"
            "请在此实现对外部 RAG 服务的知识写入调用"
            f"（endpoint={self.endpoint!r}, collection={self.collection!r}）。"
        )
