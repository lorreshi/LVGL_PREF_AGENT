# Implementation Plan: Embedded Device Agent

## Overview

本实施计划将设计（平台底座 + 可插拔能力）拆解为增量式编码任务。总体路线：先搭建项目骨架与共享数据契约（Pydantic），再自底向上实现共享底座（Config/LLM/Device/Memory 三套 Base+Factory），随后落地 Capability A 的确定性核心工具（可离线单测、属性测试主战场），接着实现 4 个 LLM 子智能体与 Orchestrator 调优闭环 subgraph，再实现平台层（BaseCapability/CapabilityFactory/Router）与 Capability B 骨架，最后接入语言无关 API 边界并做系统级闭环测试。

约定：每个子任务完成即写单元测试（Tier 1），每个章节完成即写集成测试（Tier 2），系统级闭环测试（Tier 3）在最后贯通。标 `*` 的测试子任务为可选，可为加速 MVP 跳过；核心实现子任务不标 `*`，必须实现。所有副作用经接口注入，测试时以 `FakeDeviceIO`/`FakeLLMProvider`/`InMemoryStore` 替换。语言：Python 3.11+，运行时校验用 Pydantic v2，属性测试用 hypothesis。

## Tasks

- [ ] 1. 搭建项目骨架与共享数据契约
  - [x] 1.1 初始化 Python 项目结构与工具链
    - 创建 `pyproject.toml`（Python 3.11+，依赖：langgraph、pydantic v2、pyserial、fastapi、pytest、hypothesis、mypy）
    - 建立 `src/embedded_device_agent/` 与 `tests/{harness,unit,integration,system,fixtures}/` 目录骨架及 `__init__.py`
    - 配置 pytest 与 mypy 基础设置
    - _Requirements: 12.1_

  - [x] 1.2 定义共享 Pydantic 数据契约
    - 在 `core/models.py` 实现 `ArtifactRef`、`RawTraceArtifact`、`SystraceEvent`、`FilterResult`、`CallTreeNode`、`ParseResult`、`HotspotEntry`、`SlowFrame`、`FrameReport`、`TraceQuery`、`TraceSlice`、`KnowledgeRecord`
    - 确保所有模型可独立构造、JSON 可序列化（支撑 API 边界与断言）
    - _Requirements: 3.1, 4.1, 5.1, 9.1, 15.3, 15.4_

  - [x]* 1.3 编写数据契约单元测试
    - 测试各模型的构造、校验与 JSON round-trip 序列化
    - **Property 9: 工件语言中立**（FrameReport/SlowFrame/HotspotEntry 无损 JSON 序列化且保留火焰图字段）
    - **Validates: Requirements 15.3, 15.4**

- [ ] 2. 实现配置层（Config_Loader + Pydantic 模型 + 工厂契约）
  - [x] 2.1 实现配置 Pydantic 模型
    - 在 `core/config/models.py` 实现 `LLMConfig`、`DeviceConfig`、`RetrieverConfig`、`AppConfig`（含 `frame_budget_us`、`hotspot_top_n`、`report_token_budget`、`mode`、`max_iterations`、`intervention_points`、`max_safe_baud`）
    - _Requirements: 11.1, 12.1_

  - [x] 2.2 实现 Config_Loader
    - 在 `core/config/loader.py` 实现 YAML 加载 → Pydantic 校验；缺字段/非法值在任何 agent 运行前报出具体字段与原因
    - 创建 `config/config.example.yaml` 示例
    - _Requirements: 12.1, 12.2, 11.1_

  - [ ]* 2.3 编写 Config_Loader 单元测试
    - 测试合法 YAML 装配、缺失/非法字段的具体报错定位
    - **Property 7: 配置先校验后运行**（校验未通过前不构造/执行任何组件）
    - **Validates: Requirements 12.1, 12.2**

- [ ] 3. 实现 LLM_Provider 底座（Base + Factory）
  - [x] 3.1 实现 BaseLLMProvider 与 LLMProviderFactory
    - 在 `core/llm/base.py` 定义 `BaseLLMProvider`（`get_chat_model`、`name`）
    - 在 `core/llm/factory.py` 实现基于 `type` 字段的装饰器注册与构造
    - _Requirements: 12.3_

  - [ ] 3.2 实现具体 LLM Provider 后端
    - 在 `core/llm/providers/` 实现 `AnthropicProvider`、`OpenAIProvider`、`OpenAICompatibleProvider`（读 `api_key_env`、`base_url`）
    - 注册到工厂
    - _Requirements: 12.3_

  - [ ]* 3.3 编写 LLM 工厂单元测试
    - 测试 `type` 分发构造正确子类、缺失字段报错
    - _Requirements: 12.3_

- [ ] 4. 实现 DeviceIO 底座（Base + Factory + 后端）
  - [x] 4.1 实现 DeviceIO 基类与 HumanInLoopMixin
    - 在 `core/device/base.py` 定义 `DeviceIO`（`open_serial`、`capture`、`build`、`flash`、`inject_input`、`send_cmd`）与 `HumanInLoopMixin`（不可脚本化时返回 `InterventionRequest` 而非抛错）
    - 在 `core/device/factory.py` 实现基于 `type` 的注册与构造
    - _Requirements: 14.1, 14.2, 14.3, 12.4_

  - [ ] 4.2 实现 SerialDeviceIO 后端
    - 在 `core/device/backends/serial.py` 用 pyserial 实现全自动后端；`open_serial` 前校验 `baud <= max_safe_baud`；打不开返回带端口与原因的描述性错误；`send_cmd` 发命令并返回响应
    - _Requirements: 2.1, 2.3, 2.4, 14.4_

  - [ ] 4.3 实现 HalfAutoDeviceIO 后端
    - 在 `core/device/backends/half_auto.py` 实现部分脚本化后端，其余操作产出人在环干预点
    - _Requirements: 14.2, 7.3, 11.3_

  - [ ]* 4.4 编写 DeviceIO 单元测试
    - 测试波特率超限拒绝（2.3）、串口打不开报错（2.4）、send_cmd 往返（14.4）、干预点降级（14.2）
    - _Requirements: 2.3, 2.4, 14.2, 14.4_

- [ ] 5. 实现 Memory / Retriever 底座（Base + Factory + 后端）
  - [ ] 5.1 实现 BaseRetriever 与工厂
    - 在 `core/memory/base.py` 定义 `BaseRetriever`（`recall(symptom, k)`、`persist(record)`）
    - 在 `core/memory/factory.py` 实现基于 `type` 的注册与构造
    - _Requirements: 9.4, 12.5_

  - [ ] 5.2 实现 LocalStoreRetriever（基于 LangGraph Store）
    - 在 `core/memory/backends/local_store.py` 实现语义召回与持久化；按能力标识对 Knowledge_Record 分区；使记录跨线程可用
    - 预留 `ExternalRAGRetriever` 作为 tool 调用外部 RAG 的骨架
    - _Requirements: 9.1, 9.2, 9.5, 18.3_

  - [ ]* 5.3 编写 Retriever 单元测试
    - 测试 `type` 分发、召回相关性排序、跨能力分区
    - _Requirements: 9.2, 9.4, 18.3_

- [ ] 6. 组装 Core_Services 并做底座集成测试
  - [ ] 6.1 实现 CoreServices 组装
    - 在 `core/services.py` 实现 `CoreServices` dataclass 与由 `Config_Loader` 一次性构造并注入 device/llm/retriever/checkpointer/store/config 的装配函数
    - _Requirements: 18.1, 18.4_

  - [ ]* 6.2 编写底座配置装配链集成测试（Tier 2）
    - 从一份 YAML 经 Config_Loader 构造出全套工厂实例（LLM/Device/Retriever）并组装 CoreServices
    - 记忆链：`persist(KnowledgeRecord) → recall(symptom)` 跨"线程"可召回
    - **Property 6: 知识跨线程可见**
    - **Validates: Requirements 9.5, 18.1, 18.4**

- [ ] 7. 检查点 - 确保底座测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. 实现 Capability A 确定性核心工具（可离线单测）
  - [ ] 8.1 实现 Trace_Filter
    - 在 `capabilities/perf_tuning/tools/trace_filter.py` 实现 `trace_filter(raw) -> FilterResult`：原始日志 → systrace；排除破坏配对的交错事件；报告受损区间与保留/排除计数；保留时间戳
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 8.2 编写 Trace_Filter 单元测试与属性测试
    - 单测：交错破坏配对时的排除与计数、时间戳保留
    - **Property 2: B/E 配对守恒**（retained + excluded = 输入总数；保留事件必成对）
    - **Property 8: 确定性核心无副作用**（相同输入恒等输出，无外部副作用）
    - **Validates: Requirements 3.2, 3.3, 13.2**

  - [ ] 8.3 实现 Trace_Parser
    - 在 `capabilities/perf_tuning/tools/systrace_parser.py` 实现 `parse_systrace(trace) -> ParseResult`：`B|tid|func`/`E|tid|func` → 按线程调用树；耗时 = end - begin；报告未匹配 begin 并排除；流式增量读取大文件；保留亚毫秒（微秒）精度
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [ ]* 8.4 编写 Trace_Parser 单元测试与 round-trip 属性测试
    - 单测：正常配对、缺失 end 排除、亚毫秒精度保留
    - **Property 1: 解析往返一致**（`serialize(parse(x))` 与 `x` 有序事件集等价，hypothesis）
    - **Property 3: 耗时非负且区间嵌套**（duration >= 0；子区间含于父区间）
    - **Property 4: 精度不丢失**（耗时精度不低于输入时间戳精度）
    - **Validates: Requirements 4.4, 4.2, 4.5**

  - [ ] 8.5 实现 Frame_Analyzer
    - 在 `capabilities/perf_tuning/tools/frame_analyzer.py` 实现 `analyze_frames(tree, cfg) -> FrameReport`：识别超预算慢帧；按聚合耗时排序热点并裁剪至 `hotspot_top_n`；输出聚合统计（total_frames/p95）；无慢帧分支；受 `report_token_budget` 约束超限二次聚合
    - _Requirements: 5.1, 5.2, 5.4_

  - [ ]* 8.6 编写 Frame_Analyzer 单元测试
    - 单测：超预算判定、热点排序与 top-N 裁剪、无慢帧分支、token 预算二次聚合
    - **Property 8: 确定性核心无副作用**
    - **Property 12: 原始日志不进上下文**（仅 top-N + 聚合统计进 FrameReport）
    - **Validates: Requirements 5.1, 5.2, 5.4, 13.2, 4.1**

  - [ ] 8.7 实现 query_trace 按需下钻检索工具
    - 在 `capabilities/perf_tuning/tools/frame_analyzer.py`（或独立模块）实现 `query_trace(ref, query) -> TraceSlice`：按帧号/函数/时间窗坐标返回受 `max_nodes` 限制的调用树切片
    - _Requirements: 5.2, 4.1_

  - [ ]* 8.8 编写 query_trace 单元测试
    - 测试按坐标取片、`max_nodes` 截断标记
    - **Property 12: 原始日志不进上下文**（细节经 query_trace 按需分片）
    - **Validates: Requirements 4.1, 5.2**

- [ ] 9. 章节集成测试 - 日志处理链（Tier 2）
  - [ ]* 9.1 编写日志处理链集成测试
    - 用真实样例日志端到端：`原始样例日志 → Trace_Filter → Trace_Parser → Frame_Analyzer`，断言慢帧报告与已知答案一致
    - fixtures 放入 `tests/fixtures/`
    - _Requirements: 3.1, 4.1, 5.1_

- [ ] 10. 检查点 - 确保确定性核心测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. 实现 Capability A 状态与 LLM 子智能体
  - [ ] 11.1 定义 TuningRunState
    - 在 `capabilities/perf_tuning/state.py` 实现 `TuningRunState` TypedDict（thread_id/scenario/mode/iteration/max_iterations/latest_artifact 仅存 ArtifactRef/baseline/latest/history/optimizations/recalled_knowledge/done）
    - _Requirements: 1.1, 10.1, 11.1_

  - [ ] 11.2 实现 Collector 子智能体
    - 在 `capabilities/perf_tuning/agents/collector.py` 实现：指挥 DeviceIO 采集、判断日志完整性、必要时降波特率重采、注入输入触发场景、记录采集时长与工件位置
    - _Requirements: 2.2, 2.5, 2.6_

  - [ ]* 11.3 编写 Collector 单元测试
    - 用 FakeDeviceIO 测试采集流程、降波特率重采决策、状态记录
    - _Requirements: 2.2, 2.5, 2.6_

  - [ ] 11.4 实现 Analyzer 子智能体
    - 在 `capabilities/perf_tuning/agents/analyzer.py` 实现：调用确定性工具解读慢帧/热点，产出带证据（具体函数与帧）的瓶颈概括；无慢帧时报告
    - _Requirements: 5.3, 5.4_

  - [ ]* 11.5 编写 Analyzer 单元测试
    - 用 FakeLLMProvider 测试概括产出与证据引用
    - _Requirements: 5.3_

  - [ ] 11.6 实现 Instrumentor 子智能体
    - 在 `capabilities/perf_tuning/agents/instrumentor.py` 实现：在 C 源码区域插入 `LV_PROFILER_BEGIN_TAG`/`END_TAG` 成对埋点（标签参数为字符串字面量）、触发重编译烧录、精度不足时报告需微秒级 tick 回调
    - _Requirements: 6.1, 6.2, 6.3, 6.6_

  - [ ]* 11.7 编写 Instrumentor 单元测试
    - 测试埋点插入、字符串字面量标签、微秒 tick 报告分支
    - _Requirements: 6.1, 6.2, 6.6_

  - [ ] 11.8 实现 Optimizer 子智能体
    - 在 `capabilities/perf_tuning/agents/optimizer.py` 实现：针对叶子热点提出/应用源码优化（引用目标函数与预期效果）；Full_Auto 直接应用+重编译烧录，Half_Auto 呈现待批准
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 11.9 编写 Optimizer 单元测试
    - 测试优化提议结构、Full/Half 模式分支、状态记录
    - _Requirements: 7.1, 7.3, 7.4_

- [ ] 12. 实现 Capability A Orchestrator 与调优闭环 subgraph
  - [ ] 12.1 实现 Orchestrator 工具装配
    - 在 `capabilities/perf_tuning/orchestrator.py` 用 `create_agent` 构建：子智能体作为工具（collector/analyzer/instrumentor/optimizer）、确定性操作作为普通工具（filter/parser/frame_analyzer/query_trace）、记忆工具（recall/persist）；工具返回后写入 state 再决策；工具报错按模式决定下一步
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [ ]* 12.2 编写 Orchestrator 工具装配单元测试
    - 验证子智能体 vs 普通工具区分、返回结果写入 state、错误处理路径
    - **Property 8: 确定性核心无副作用**（确定性工具作为普通工具挂载）
    - **Validates: Requirements 13.1, 13.2, 13.3, 13.4**

  - [ ] 12.3 实现调优闭环 subgraph（intake→recall→plan→collect→filter_parse→analyze→decide→instrument/optimize→memorize/halt）
    - 在 `capabilities/perf_tuning/graph.py` 用 LangGraph 构建状态图；intake 创建 Tuning_Run 并记录场景（缺场景则先请求）；recall 语义召回历史并纳入上下文；decide 每轮 `iteration++` 并对照 `Max_Iteration_Limit`；instrument/optimize 后回到 collect 复测；接入 Checkpointer 使可恢复
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 6.4, 6.5, 8.1, 8.2, 8.3, 8.4, 9.3, 10.1, 10.2, 11.2, 11.4, 11.5_

  - [ ] 12.4 实现 Half_Auto 人在环中断与验证/知识沉淀
    - 在 `graph.py` 于配置干预点用 `interrupt` 暂停（状态由 Checkpointer 保留）；优化后重采重析对比基线、无改进标记无效并回退替代方案；验证为改进则持久化 Knowledge_Record
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 9.1, 10.3, 11.3_

  - [ ]* 12.5 编写调优闭环单元测试
    - 用 FakeDeviceIO+FakeLLMProvider 测试 decide 分支、迭代计数、halt 触发、interrupt/resume
    - **Property 5: 迭代有界终止**（迭代不超过 Max_Iteration_Limit，有限步到达 memorize/halt）
    - **Validates: Requirements 6.5, 11.4, 11.5**

- [ ] 13. 检查点 - 确保 Capability A 测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. 实现平台层（BaseCapability / CapabilityFactory / Router）
  - [ ] 14.1 实现 BaseCapability 与 CapabilityFactory
    - 在 `capabilities/base.py` 定义 `BaseCapability`（name/description/build_graph/as_tool）
    - 在 `capabilities/factory.py` 实现装饰器注册表与 `create(key, core)`
    - _Requirements: 16.1, 16.2, 16.4_

  - [ ] 14.2 将 PerfTuningCapability 封装并注册
    - 在 `capabilities/perf_tuning/capability.py` 实现 `PerfTuningCapability(BaseCapability)`，封装闭环 subgraph 并 `@register("perf_tuning")`；实现 `as_tool()` 暴露为 subgraph-as-tool
    - _Requirements: 16.4, 16.5_

  - [ ] 14.3 实现 Router_Orchestrator
    - 在 `router.py` 用 `create_agent` 将各注册 Capability 的 `as_tool()` 装配为工具，依 `description` 判定意图分发；无匹配返回可用能力清单；多匹配请求澄清；能力完成后返回结果
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 16.3_

  - [ ]* 14.4 编写平台层单元测试
    - 测试注册即可路由、意图分发、无/多匹配处理
    - **Property 10: 能力注册即可路由**（注册即纳入分发候选，无需改 Router/Core 代码）
    - **Validates: Requirements 16.3, 16.4, 17.1, 17.2, 17.3**

- [ ] 15. 实现 Capability B（FunctionalTest）契约与骨架
  - [ ] 15.1 定义 FunctionalTest 数据契约
    - 在 `capabilities/functional_test/models.py` 定义测试用例、断言、测试报告的 Pydantic 契约（API 边界可语言中立暴露）
    - _Requirements: 19.3_

  - [ ] 15.2 实现 FunctionalTestCapability 骨架与最小可跑流程
    - 在 `capabilities/functional_test/capability.py` 实现 `FunctionalTestCapability(BaseCapability)` 并 `@register("functional_test")`
    - 在 `capabilities/functional_test/graph.py` 实现经共享 DeviceIO（inject_input/send_cmd/capture）驱动设备的最小可端到端跑通骨架
    - _Requirements: 19.1, 19.2, 19.4, 18.2_

  - [ ]* 15.3 编写 Capability B 骨架测试
    - 在 Test_Harness 上跑通最小测试流程，证明平台契约成立
    - **Property 10: 能力注册即可路由**（第二能力验证扩展性）
    - **Validates: Requirements 19.1, 19.4, 18.2**

- [ ] 16. 实现测试 Harness
  - [ ] 16.1 实现 FakeDeviceIO 与 FakeLLMProvider
    - 在 `tests/harness/` 实现 `FakeDeviceIO`（实现 DeviceIO 接口，回放录制采集日志与构建/烧录结果）与 `FakeLLMProvider`（实现 BaseLLMProvider 接口，回放预置决策）
    - 录制 fixtures 放入 `tests/fixtures/`
    - _Requirements: 20.1, 20.2, 20.4_

  - [ ]* 16.2 编写 Harness 决定性替换测试
    - 验证经注入 fake 即可驱动任意能力完整 subgraph，能力代码零改动，不触达真实硬件/网络/LLM
    - **Property 11: Harness 决定性替换**
    - **Validates: Requirements 20.3, 20.4**

- [ ] 17. 实现语言无关 API 边界
  - [ ] 17.1 实现 FastAPI DTO 与路由
    - 在 `api/dto.py` 定义 JSON 可序列化 DTO；在 `api/app.py` 实现 `POST /runs`（返回 thread_id）、`GET /runs/{id}`、`GET /runs/{id}/report`（含火焰图字段）、`GET /knowledge`、`POST /runs/{id}/resume`
    - 引用不存在的 run/record 返回带缺失标识符的错误
    - **安全提示**：API 默认无鉴权，若对外暴露需另加访问控制
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_

  - [ ]* 17.2 编写 API 边界集成测试（Tier 2）
    - `POST /runs` 返回 thread_id；`GET /runs/{id}/report` 返回含火焰图字段 JSON；引用不存在标识符报错
    - **Property 9: 工件语言中立**
    - **Validates: Requirements 15.3, 15.4, 15.5, 15.6**

- [ ] 18. 系统级闭环测试（Tier 3）
  - [ ]* 18.1 编写全自动闭环系统测试
    - 用录制 fixture 驱动整张 LangGraph 图走通 happy path：采集→分析→下钻→优化→验证→沉淀，并落一条 Knowledge_Record
    - **Property 6: 知识跨线程可见**
    - **Validates: Requirements 9.5, 11.2**

  - [ ]* 18.2 编写迭代上限守卫系统测试
    - 构造永不达标 fixture，断言在 Max_Iteration_Limit 处 halt 并报告最佳证据
    - **Property 5: 迭代有界终止**
    - **Validates: Requirements 6.5, 11.5**

  - [ ]* 18.3 编写人在环恢复系统测试
    - Half_Auto 在干预点 interrupt，resume 后 Checkpointer 恢复状态继续
    - **Validates: Requirements 10.2, 10.3, 11.3**

- [ ] 19. 最终检查点 - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- 标 `*` 的子任务为可选测试，可为加速 MVP 跳过；核心实现子任务必须实现。
- 每个子任务引用具体需求编号以保证可追溯。
- 属性测试对应 design.md 的 Correctness Properties（Property 1–12），就近置于对应实现之后以尽早捕获错误。
- 检查点用于增量验证；三级测试策略（Tier 1 单元 / Tier 2 集成 / Tier 3 系统）贯穿始终。
- 所有副作用经接口注入，测试时以 FakeDeviceIO/FakeLLMProvider/InMemoryStore 替换。

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["1.3", "2.2", "3.1", "4.1", "5.1"] },
    { "id": 3, "tasks": ["2.3", "3.2", "4.2", "4.3", "5.2", "8.1", "8.3", "8.5", "11.1"] },
    { "id": 4, "tasks": ["3.3", "4.4", "5.3", "8.2", "8.4", "8.6", "8.7", "6.1"] },
    { "id": 5, "tasks": ["6.2", "8.8", "9.1", "11.2", "11.4", "11.6", "11.8"] },
    { "id": 6, "tasks": ["11.3", "11.5", "11.7", "11.9", "12.1"] },
    { "id": 7, "tasks": ["12.2", "12.3"] },
    { "id": 8, "tasks": ["12.4", "14.1"] },
    { "id": 9, "tasks": ["12.5", "14.2", "15.1"] },
    { "id": 10, "tasks": ["14.3", "15.2", "16.1"] },
    { "id": 11, "tasks": ["14.4", "15.3", "16.2", "17.1"] },
    { "id": 12, "tasks": ["17.2", "18.1", "18.2", "18.3"] }
  ]
}
```
