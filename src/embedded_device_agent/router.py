"""Router_Orchestrator：依意图把请求分发到各可插拔能力（任务 14.3）。

严格对照 design.md「路由层 → Router_Orchestrator」与需求 17：用 LangGraph 预置的
ReAct 主体（``create_react_agent``，即设计所述 ``create_agent``）构建，把**每个已注册
Capability 的 ``as_tool()``** 装配为工具（subgraph-as-tool），依各能力的 ``description``
判定意图并分发（需求 17.1）；无匹配返回可用能力清单（17.2），多匹配请求澄清（17.3），
能力完成后回传其结果（17.4）。

关键：能力候选来自 :class:`CapabilityFactory` 的注册表——任一遵循 ``BaseCapability``
契约并 ``@register`` 的能力**无需修改 Router/核心代码**即被纳入分发候选（需求 16.3，
Property 10）。

可测试性：``create_react_agent`` 在构建期把工具绑定到模型，需要具备 tool-calling
能力的模型；因此本模块把「候选枚举」与「工具装配」拆为可离线断言的
:func:`available_capabilities` / :func:`build_capability_tools`，Router 主体则由
:func:`build_router` 组装（Property 10 的路由候选可脱离真实 LLM 验证）。
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from embedded_device_agent.capabilities.factory import CapabilityFactory
from embedded_device_agent.core.services import CoreServices

__all__ = [
    "available_capabilities",
    "build_capability_tools",
    "build_router",
    "ROUTER_SYSTEM_PROMPT",
]


ROUTER_SYSTEM_PROMPT = (
    "你是嵌入式设备智能体平台的路由器（Router）。你的职责是理解用户意图，并把请求"
    "分发给最合适的能力工具。可用能力如下：\n"
    "{capabilities}\n\n"
    "分发规则：\n"
    "1) 依据每个能力的描述判断意图，选择**唯一**最匹配的能力工具并调用它；\n"
    "2) 若没有任何能力匹配用户意图，不要臆测——直接回复当前可用的能力清单，"
    "请用户在其中选择；\n"
    "3) 若有多个能力都可能匹配，向用户澄清其真实意图后再分发；\n"
    "4) 能力工具执行完成后，把其结果如实回传给用户。"
)


def available_capabilities() -> list[dict[str, str]]:
    """枚举全部已注册能力的 ``{key, name, description}``（需求 17.2）。

    候选直接取自 :class:`CapabilityFactory` 注册表——新增注册能力即自动出现，无需改动
    本函数或 Router（需求 16.3，Property 10）。
    """
    capabilities: list[dict[str, str]] = []
    for key in CapabilityFactory.available():
        cap_cls = CapabilityFactory._registry[key]
        capabilities.append(
            {
                "key": key,
                "name": getattr(cap_cls, "name", key) or key,
                "description": getattr(cap_cls, "description", "") or "",
            }
        )
    return capabilities


def build_capability_tools(core: CoreServices) -> list[BaseTool]:
    """把每个已注册能力的 ``as_tool(core)`` 装配为 Router 的工具集（需求 16.4/17.1）。

    遍历 :class:`CapabilityFactory` 注册表构造各能力并取其 subgraph-as-tool；
    注册即入选，无需改动 Router（Property 10）。
    """
    tools: list[BaseTool] = []
    for key in CapabilityFactory.available():
        capability = CapabilityFactory.create(key, core)
        tools.append(capability.as_tool(core))
    return tools


def _render_capability_catalog() -> str:
    """把可用能力清单渲染为供系统提示注入的多行文本。"""
    lines = [f"- {c['name']}：{c['description']}" for c in available_capabilities()]
    return "\n".join(lines) if lines else "-（当前无已注册能力）"


def build_router(core: CoreServices, *, name: str = "router_orchestrator") -> CompiledStateGraph:
    """用 ``create_react_agent`` 装配 Router 主体（需求 17）。

    把全部已注册能力的 ``as_tool()`` 作为工具挂载，并注入描述分发规则与可用能力清单的
    系统提示。返回已编译的可运行主体（注入共享 Checkpointer/Store）。
    """
    tools = build_capability_tools(core)

    def _prompt(state) -> list[AnyMessage]:  # noqa: ANN001 - LangGraph 状态字典
        system = SystemMessage(
            content=ROUTER_SYSTEM_PROMPT.format(capabilities=_render_capability_catalog())
        )
        return [system, *state.get("messages", [])]

    return create_react_agent(
        core.llm.get_chat_model(),
        tools,
        prompt=_prompt,
        checkpointer=core.checkpointer,
        store=core.store,
        name=name,
    )
