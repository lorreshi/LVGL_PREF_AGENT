"""query_trace：调用树按需下钻检索工具（任务 8.7）。

作为**确定性普通工具**由 Orchestrator 挂载、供 LLM 调用（需求 13.2）。它是
「LLM 不读日志、只按需翻页」的落点（design.md「大日志上下文预算」机制 3）：
LLM 平时只见到受 ``report_token_budget`` 约束的 ``FrameReport`` 摘要
（Property 12），需要细节时再以坐标（帧号 / 函数 / 线程 / 时间窗）调用本工具，
按需取回**某一小段**调用树切片，使 context 只按需增长，原始日志始终留在磁盘。

纯输入输出：给定相同的 artifact 内容与相同的 ``TraceQuery``，恒产生相同的
``TraceSlice``，不产生任何外部副作用（Property 8）。

坐标语义（``TraceQuery``，多项同时给出时取交集 / AND）：

* ``frame_index``：帧序下标，基准与 ``Frame_Analyzer`` 完全一致——即
  ``derive_frames`` 产出的（begin_us, tid, func）稳定排序帧序（亦是
  ``SlowFrame.index``）。越界返回空切片。
* ``func``：函数名；从候选帧内**自顶向下**定位首个同名节点，取其整棵子树。
* ``tid``：线程号；限定候选帧 / 节点所属线程。
* ``time_window_us``：``(lo, hi)`` 闭区间（微秒）；定位**完全落入**该窗口的
  最上层节点（据此在帧内进一步收窄，而非返回整帧）。

返回 ``TraceSlice``：命中子树按前序遍历、以 ``max_nodes`` 为总节点预算裁剪；
一旦有节点因预算被丢弃即置 ``truncated=True``（需求 5.2 的上下文预算约束）。

需求对照：需求 4.1（按线程调用树的坐标检索）、需求 5.2（top-N 之外的细节经
按需分片获取，不整体注入 context）。
"""

from __future__ import annotations

from embedded_device_agent.capabilities.perf_tuning.tools.frame_analyzer import (
    derive_frames,
)
from embedded_device_agent.capabilities.perf_tuning.tools.systrace_parser import (
    parse_systrace,
)
from embedded_device_agent.core.models import (
    ArtifactRef,
    CallTreeNode,
    ParseResult,
    TraceQuery,
    TraceSlice,
)

__all__ = ["query_trace"]


def _node_matches(node: CallTreeNode, query: TraceQuery) -> bool:
    """判定某节点是否满足全部**已给出**的坐标谓词（AND 语义）。

    ``frame_index`` 属帧级选择、不在此判定；此处只判 ``func`` / ``tid`` /
    ``time_window_us``（后者要求节点区间**完全落入**窗口，以便在帧内收窄）。
    """
    if query.func is not None and node.func != query.func:
        return False
    if query.tid is not None and node.tid != query.tid:
        return False
    if query.time_window_us is not None:
        lo, hi = query.time_window_us
        if node.begin_us < lo or node.end_us > hi:
            return False
    return True


def _has_narrowing_predicate(query: TraceQuery) -> bool:
    """是否存在需要在帧内**继续下钻**的收窄谓词（func / 时间窗）。

    ``tid`` 仅作帧级过滤、不驱动帧内下钻；``frame_index`` 亦属帧级选择。
    """
    return query.func is not None or query.time_window_us is not None


def _collect_matches(node: CallTreeNode, query: TraceQuery) -> list[CallTreeNode]:
    """前序遍历，收集满足谓词的**最上层**节点（命中即取整棵子树、不再深入）。

    未命中则继续向子节点递归，从而在帧内精确定位收窄坐标对应的一小段。
    """
    if _node_matches(node, query):
        return [node]
    matches: list[CallTreeNode] = []
    for child in node.children:
        matches.extend(_collect_matches(child, query))
    return matches


def _select_roots(frames: list[CallTreeNode], query: TraceQuery) -> list[CallTreeNode]:
    """依坐标选出候选帧（``frame_index`` 帧级选择 + ``tid`` 帧级过滤）。"""
    if query.frame_index is not None:
        if 0 <= query.frame_index < len(frames):
            candidates = [frames[query.frame_index]]
        else:
            candidates = []
    else:
        candidates = list(frames)
    if query.tid is not None:
        candidates = [f for f in candidates if f.tid == query.tid]
    return candidates


def _prune(
    nodes: list[CallTreeNode], budget: list[int]
) -> tuple[list[CallTreeNode], bool]:
    """按前序遍历、以 ``budget`` 为总节点预算裁剪节点树，返回（副本, 是否截断）。

    ``budget`` 为单元素可变列表（跨递归共享的剩余额度）。每纳入一个节点扣 1；
    额度耗尽即停止纳入其余兄弟 / 子节点并标记截断。返回的是**深拷贝副本**，
    避免改动入参（保持纯函数）。
    """
    result: list[CallTreeNode] = []
    truncated = False
    for node in nodes:
        if budget[0] <= 0:
            truncated = True
            break
        budget[0] -= 1
        children, child_truncated = _prune(node.children, budget)
        result.append(node.model_copy(update={"children": children}))
        truncated = truncated or child_truncated
    return result, truncated


def query_trace(
    ref: ArtifactRef | ParseResult,
    query: TraceQuery,
) -> TraceSlice:
    """按坐标从调用树 artifact 下钻取回一小段切片（需求 4.1 / 5.2）。

    Args:
        ref: 指向 systrace / 调用树 artifact 的 ``ArtifactRef``（据此重新解析、
            确定性地重建按线程调用树）；亦接受已就绪的 ``ParseResult``（便于
            单测与免磁盘往返）。
        query: 下钻坐标（帧号 / 函数 / 线程 / 时间窗）与 ``max_nodes`` 上限。

    Returns:
        ``TraceSlice``：命中子树按前序裁剪至 ``max_nodes`` 个节点；
        ``truncated`` 标记是否因预算丢弃了节点。

    不变量：相同 artifact 内容与相同 query 恒产生相同切片且无副作用
    （Property 8）；LLM 仅按需获取细节切片，原始日志绝不整体进入 context
    （Property 12）。
    """
    tree = ref if isinstance(ref, ParseResult) else parse_systrace(ref)

    # 帧序基准与 Frame_Analyzer 完全一致（单一真源），故坐标 frame_index 通用。
    frames = derive_frames(tree)
    candidate_roots = _select_roots(frames, query)

    # 存在收窄谓词（func / 时间窗）时在候选帧内继续下钻定位最上层命中节点；
    # 否则候选帧本身即命中子树（仅帧号 / 线程坐标）。
    if _has_narrowing_predicate(query):
        selected: list[CallTreeNode] = []
        for root in candidate_roots:
            selected.extend(_collect_matches(root, query))
    else:
        selected = candidate_roots

    # max_nodes <= 0 视为不返回任何节点（并据实际是否有命中标记截断）。
    budget = max(0, query.max_nodes)
    nodes, truncated = _prune(selected, [budget])
    return TraceSlice(nodes=nodes, truncated=truncated)
