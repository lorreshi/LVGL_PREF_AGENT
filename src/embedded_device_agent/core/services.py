"""Core_Services 组装：共享底座的一次性构造与注入（需求 18.1 / 18.4）。

严格对照 design.md "平台层 → Core_Services" 章节：

``CoreServices`` 是被**所有能力复用**的共享底座——设备控制（DeviceIO）、
LLM（LLM_Provider）、长期记忆检索（Retriever）、短期/长期记忆基础设施
（Checkpointer + Store）与配置（AppConfig）。它由平台在启动阶段**构造一次**
并注入各 Capability，而非由各能力自行实例化（需求 18.4）。

装配语义（``build_core_services``）：

* **验证先于运行**（需求 12）：入参可为 YAML 配置路径或已构造的
  :class:`AppConfig`。为路径时经 :class:`ConfigLoader` 加载并做 Pydantic 校验，
  任何缺字段 / 非法值都会在任何 agent 运行前抛出 :class:`ConfigError`。
* **一次性构造工厂实例**：依据校验后的配置，经 ``LLMProviderFactory`` /
  ``DeviceIOFactory`` / ``RetrieverFactory`` 各构造一次实例，统一注入
  ``CoreServices``（需求 18.1）。
* **Retriever 与 Store 共享同一实例**（需求 9.5 / 18）：默认 Retriever 后端
  ``local_store`` 直接注入平台持有的同一 :class:`BaseStore`，使任一线程
  ``persist`` 的知识对后续任意 ``thread_id`` 的 ``recall`` 可见（跨线程可见）。
* **默认内存后端**：Checkpointer 默认 :class:`InMemorySaver`、Store 默认
  :class:`InMemoryStore`（LangGraph 本地起步）；两者均可注入以便测试或替换为
  持久化后端。

依赖注入友好：``checkpointer`` / ``store`` 均可显式注入；不注入时使用内存默认值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from embedded_device_agent.core.config.loader import ConfigLoader
from embedded_device_agent.core.config.models import AppConfig, RetrieverConfig

# 导入三套底座**包**（而非仅工厂模块）以触发具体后端经装饰器注册到各自工厂：
#   - llm.providers: anthropic / openai / openai_compatible
#   - device.backends: serial / half_auto
#   - memory.backends: local_store / external_rag
# 若仅导入 factory 模块，注册表将为空，create() 会因"未知 type"报错。
from embedded_device_agent.core.device import DeviceIO, DeviceIOFactory
from embedded_device_agent.core.llm import BaseLLMProvider, LLMProviderFactory
from embedded_device_agent.core.memory import BaseRetriever, RetrieverFactory
from embedded_device_agent.core.memory.backends.local_store import LocalStoreRetriever

__all__ = ["CoreServices", "build_core_services"]


@dataclass
class CoreServices:
    """被所有 Capability 复用的共享底座（需求 18.1）。

    由平台经 :func:`build_core_services` 构造一次并注入各能力（需求 18.4）；
    各字段分别对应设计中的一件基础设施：

    * ``device``：设备控制面（串口 / 编译 / 烧录 / 输入注入 / 命令往返）。
    * ``llm``：LLM 提供方（chat model 封装）。
    * ``retriever``：长期记忆检索面（语义召回 / 知识持久化）。
    * ``checkpointer``：LangGraph 线程范围短期状态（可恢复）。
    * ``store``：LangGraph 跨线程长期记忆面；本地 Retriever 与其共享同一实例。
    * ``config``：校验后的顶层 :class:`AppConfig`。
    """

    device: DeviceIO
    llm: BaseLLMProvider
    retriever: BaseRetriever
    checkpointer: BaseCheckpointSaver
    store: BaseStore
    config: AppConfig


def _build_retriever(cfg: RetrieverConfig, store: BaseStore) -> BaseRetriever:
    """构造 Retriever，并让本地后端与平台 Store 共享同一实例。

    默认后端 ``local_store`` 的语义召回 / 持久化都发生在 LangGraph Store 上；
    为保证「任一线程写入的知识对后续任意线程可见」（需求 9.5 / 18），此处把
    平台持有的同一 ``store`` 注入 :class:`LocalStoreRetriever`，而非让其自建
    一个独立的 Store。其余后端（如 ``external_rag`` 直连外部 RAG，不使用本地
    Store）仍经工厂按 ``type`` 分发构造，保持可插拔（需求 12.5）。
    """

    if cfg.type == "local_store":
        return LocalStoreRetriever(cfg, store=store)
    return RetrieverFactory.create(cfg)


def build_core_services(
    config: str | Path | AppConfig,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CoreServices:
    """由配置一次性装配并返回接好线的 :class:`CoreServices`（需求 18.1 / 18.4）。

    Parameters
    ----------
    config:
        YAML 配置文件路径（``str`` / :class:`~pathlib.Path`），或已构造好的
        :class:`AppConfig`。为路径时经 :class:`ConfigLoader` 加载并做 Pydantic
        校验——验证先于运行，任何缺字段 / 非法值都会在此抛出
        :class:`~embedded_device_agent.core.config.loader.ConfigError`（需求 12）。
    checkpointer:
        可选注入的 LangGraph Checkpointer；缺省使用 :class:`InMemorySaver`。
    store:
        可选注入的 LangGraph Store；缺省使用 :class:`InMemoryStore`。本地
        Retriever 后端将与该 Store 共享同一实例以保证跨线程可见（需求 9.5）。

    Returns
    -------
    CoreServices
        设备 / LLM / Retriever / Checkpointer / Store / Config 均注入完毕、
        可直接下发给各 Capability 的共享底座。
    """

    # 1) 验证先于运行：路径 → 加载并校验；已是 AppConfig 则直接采用（需求 12）。
    app_config = config if isinstance(config, AppConfig) else ConfigLoader.from_file(config)

    # 2) 默认内存后端（LangGraph 本地起步）；均可注入以便测试 / 替换持久化后端。
    checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
    store = store if store is not None else InMemoryStore()

    # 3) 依校验后的配置各构造一次工厂实例，统一注入（需求 18.1）。
    device = DeviceIOFactory.create(app_config.device)
    llm = LLMProviderFactory.create(app_config.llm)
    retriever = _build_retriever(app_config.retriever, store)

    return CoreServices(
        device=device,
        llm=llm,
        retriever=retriever,
        checkpointer=checkpointer,
        store=store,
        config=app_config,
    )
