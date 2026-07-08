# Requirements Document

## Introduction

嵌入式设备智能体平台（Embedded Device Agent）是一个基于 LangGraph（Python）构建的、由真实 LLM API 驱动的多智能体**平台**。它并非单一用途工具，而是一个共享统一底座（设备控制、LLM 提供方、两层记忆、配置、API 边界、测试 harness）、其上承载多个**可插拔能力（Capability）**的框架。首个能力是 **Capability A：LVGL 性能调优**；第二个能力是 **Capability B：嵌入式自动化测试**（本期完成契约与插件骨架，作为验证平台扩展性的第二条能力）；平台设计为可继续挂载 Capability C、D……

**Capability A（LVGL 性能调优）** 端到端自动化如下闭环：如今嵌入式开发者需手工插入 LVGL trace 埋点、通过串口抓取 profiler 日志、清洗日志、导入 Perfetto、靠肉眼观察函数耗时定位帧率卡顿。本能力自动化该闭环——采集、清洗、分析、通过更细粒度的埋点下钻、提出并应用优化、验证改进效果——同时积累可持久化的知识，使系统越用越强。

**平台架构**：一个顶层 Router_Orchestrator 依据用户意图将请求分发到对应 Capability；每个 Capability 是自带 subgraph、子智能体、工具与状态的独立插件模块，采用 LangGraph 的 subagents-as-tools（子智能体即工具）/ subgraph-as-tool 模式组织。Capability A 内部以 Orchestrator 协调 Collector、Analyzer、Instrumentor、Optimizer 等专职子智能体，并由两层记忆子系统支撑（线程范围的 Checkpointer 保存可恢复的运行状态，跨线程的 Store 保存长期知识）。确定性的重活（systrace 解析、慢帧与热点检测）以普通工具实现；只有推理步骤才作为 LLM 子智能体。

整个平台通过 YAML 配置，并在 LLM 提供方、DeviceIO、RAG/检索后端、分析器以及 Capability 注册上一致地使用 Factory + Base-class（工厂 + 基类）模式，实现"换模型/换设备/换 RAG/加能力不改核心代码"。共享底座中的 DeviceIO 抽象被所有能力复用——性能调优与自动化测试共用同一套设备控制面。平台提供确定性测试 harness（可回放的 FakeDeviceIO + FakeLLMProvider + fixtures），使任意能力都能脱离真实硬件与真实 LLM 离线、可重复地自测。系统同时支持全自动闭环模式与半自动的人在环模式，并受最大迭代次数限制约束，以防止失控和 token 消耗失控。

## Glossary

- **Agent_Platform（智能体平台）**：承载多个可插拔能力、并为其提供共享底座（DeviceIO、LLM_Provider、记忆、配置、API、测试 harness）的整体框架。
- **Capability（能力）**：平台上的一个可插拔功能模块，自带 subgraph、子智能体、工具与状态；如 Capability A（性能调优）、Capability B（自动化测试）。遵循 BaseCapability 基类 + 工厂注册。
- **Router_Orchestrator（路由编排器）**：顶层主智能体，依据用户意图选择并分发到对应 Capability，将各 Capability 作为工具（subgraph-as-tool）调用。
- **Capability_Factory（能力工厂）**：依据注册键构造 Capability 实例的工厂，支持零改动扩展新能力。
- **Core_Services（共享底座服务）**：被所有 Capability 复用的基础设施集合——DeviceIO、LLM_Provider、Retriever、Memory（Checkpointer + Store）、Config_Loader、API 边界、测试 harness。
- **PerfTuning_Capability（性能调优能力，Capability A）**：自动化 LVGL 性能调优闭环的能力模块。
- **FunctionalTest_Capability（自动化测试能力，Capability B）**：驱动嵌入式功能/自动化测试的能力模块；本期交付其契约与插件骨架。
- **Test_Harness（测试 harness）**：以可回放的 FakeDeviceIO + FakeLLMProvider + 录制 fixtures 组成的确定性试验台，使能力可脱离真实硬件与 LLM 离线、可重复自测。
- **Orchestrator（编排器）**：某个 Capability 内部接收任务、规划闭环、并协调作为工具暴露的各专职子智能体的 LangGraph 主智能体（Capability A 的编排器负责调优闭环）。
- **Collector（采集器）**：负责通过串口从目标设备抓取原始 LVGL profiler trace 日志的子智能体。
- **Analyzer（分析器）**：解读已解析的 trace 数据，识别慢帧、热点函数与可能根因的子智能体。
- **Instrumentor（埋点器）**：向 C 源码中加入更细粒度 LVGL profiler 埋点，随后触发重新编译、重新烧录与重新测试的子智能体。
- **Optimizer（优化器）**：提出并应用源码级性能优化的子智能体。
- **Trace_Filter（trace 过滤器）**：将原始 LVGL profiler 日志转换为 Android systrace 格式的确定性工具（封装 LVGL 的 `trace_filter.py`）。
- **Trace_Parser（trace 解析器）**：将带时间戳的 Android systrace `B|tid|func` / `E|tid|func` 事件解析为按线程组织、带函数耗时的调用树的确定性工具。
- **Frame_Analyzer（帧分析器）**：从已解析调用树中检测慢帧并计算热点函数的确定性工具。
- **DeviceIO（设备 IO）**：提供设备操作（open_serial、capture、build、flash、inject_input、send_cmd）的抽象层（基类 + 工厂）。
- **LLM_Provider（LLM 提供方）**：为智能体提供 LLM API 访问、通过 YAML 选择与配置的抽象（基类 + 工厂）。
- **Retriever（检索器）**：知识检索抽象（BaseRetriever 接口），含一个基于本地 Store 的实现以及一个可选的外部 RAG 实现。
- **Memory_Store（记忆存储）**：跨线程的 LangGraph Store，持久化长期性能知识记录（症状、根因、优化、效果）并支持语义召回。
- **Checkpointer（检查点存储）**：线程范围的 LangGraph checkpointer，持久化短期运行状态以使调优运行可恢复。
- **Config_Loader（配置加载器）**：加载并校验驱动整个系统的 YAML 配置的组件。
- **Knowledge_Record（知识记录）**：一条存储的长期记忆条目，将某个症状关联到其根因、已应用的优化以及实测效果。
- **Tuning_Run（调优运行）**：针对一个用户报告的性能问题的单次端到端执行实例，由一个 thread id 标识。
- **Full_Auto_Mode（全自动模式）**：闭环无需人工干预即可运行的工作模式（编译、烧录与输入注入均可脚本化）。
- **Half_Auto_Mode（半自动模式）**：系统在既定干预点暂停，由开发者执行不可脚本化步骤的工作模式。
- **Max_Iteration_Limit（最大迭代次数限制）**：单个 Tuning_Run 内下钻/优化迭代次数的配置上限。
- **Systrace_Event（systrace 事件）**：形如 `B|tid|func_name`（开始）或 `E|tid|func_name`（结束）、并附带时间戳的单条 profiler 日志记录。

## Requirements

> **需求分组**：需求 1–15 属于 **Capability A（LVGL 性能调优）**，本期完整实现；需求 16–20 属于**平台层与 Capability B（自动化测试）**——其中平台可插拔能力架构、Router、共享底座、测试 harness 本期实现，Capability B（自动化测试）本期交付契约与插件骨架、完整业务逻辑留作后续能力 spec。

### 需求 1：为某个 UI 场景上报性能问题

**用户故事：** 作为一名嵌入式开发者，我希望描述与特定 UI 场景相关的帧率或卡顿问题，以便智能体知道要诊断和调优什么。

#### 验收标准

1. 当开发者提交一个指明了某个 UI 场景的问题描述时，THE Orchestrator SHALL 创建一个由唯一 thread id 标识的 Tuning_Run，并在运行状态中记录该场景描述。
2. 如果提交的问题描述未指明任何 UI 场景，那么 THE Orchestrator SHALL 在启动调优闭环之前请求缺失的场景信息。
3. 当一个 Tuning_Run 被创建时，THE Orchestrator SHALL 向 Memory_Store 查询其症状与所报告问题相匹配的 Knowledge_Record，并 SHALL 将任何匹配项纳入运行上下文。
4. 当一个 Tuning_Run 被创建时，THE Orchestrator SHALL 生成一份将采集、分析、下钻、优化与验证各阶段排序的计划。

### 需求 2：通过串口采集 LVGL profiler trace 日志

**用户故事：** 作为一名嵌入式开发者，我希望智能体能自动通过串口从我的设备采集 profiler trace 日志，这样我就不必再手工抓取日志。

#### 验收标准

1. 当 Collector 被指派采集某个场景时，THE DeviceIO SHALL 使用 YAML 配置中定义的端口和波特率打开串口连接。
2. 当串口连接已打开且该场景被触发执行时，THE Collector SHALL 将原始 LVGL profiler trace 输出采集到一个与该 Tuning_Run 关联的原始日志工件中。
3. 如果所配置的波特率超过 YAML 配置中为该目标声明为安全的最大波特率，那么 THE DeviceIO SHALL 拒绝本次采集，并在打开连接之前报告波特率风险。
4. 如果串口连接无法打开，那么 THE DeviceIO SHALL 返回一条描述性错误，标明端口与失败原因。
5. 在目标设备需要通过输入注入来触发场景的情况下，THE DeviceIO SHALL 在采集期间注入所配置的触摸或按键输入以触发执行该场景。
6. 当一次采集完成时，THE Collector SHALL 在运行状态中记录采集时长与原始日志工件的位置。

### 需求 3：清洗原始日志并转换为 systrace 格式

**用户故事：** 作为一名嵌入式开发者，我希望原始 profiler 日志能被自动清洗并转换为 systrace 格式，以便它们可直接用于分析。

#### 验收标准

1. 当一个原始日志工件可用时，THE Trace_Filter SHALL 将该原始日志转换为 Android systrace 格式，并产出一个已清洗的 trace 工件。
2. 如果原始日志包含来自其他线程、破坏了 begin/end 配对的交错输出，那么 THE Trace_Filter SHALL 报告受损区域，并将未配对的事件从已清洗 trace 工件中排除。
3. 当转换完成时，THE Trace_Filter SHALL 报告被保留的 Systrace_Event 数量与被排除的事件数量。
4. THE Trace_Filter SHALL 在已清洗 trace 工件中保留每个被保留 Systrace_Event 的时间戳。

### 需求 4：将 systrace 解析为调用树

**用户故事：** 作为一名嵌入式开发者，我希望已清洗的 trace 被解析为带耗时的结构化调用树，以便耗时分析精确且可重复。

#### 验收标准

1. 当提供了一个已清洗 trace 工件时，THE Trace_Parser SHALL 将每个形如 `B|tid|func_name` 与 `E|tid|func_name` 的 Systrace_Event 解析为按线程组织的调用树。
2. 当为某个函数解析出一对匹配的 begin 与 end 时，THE Trace_Parser SHALL 将该函数耗时计算为 end 时间戳与 begin 时间戳之差。
3. 如果某个 begin 事件在其所属线程内没有匹配的 end 事件，那么 THE Trace_Parser SHALL 报告该未匹配事件，并将其从调用树中排除。
4. 对于所有已清洗 trace 工件，将 trace 解析为调用树、再将该调用树序列化回 systrace，SHALL 重现一组等价且有序的 Systrace_Event（往返/round-trip 属性）。
5. 在 trace 时间戳带有亚毫秒精度的情况下，THE Trace_Parser SHALL 在所计算的耗时中保留全部可用精度。

### 需求 5：检测慢帧与热点函数

**用户故事：** 作为一名嵌入式开发者，我希望智能体能定位慢帧以及在其中占据主要耗时的函数，以便我知道瓶颈在哪里。

#### 验收标准

1. 当提供了一棵调用树时，THE Frame_Analyzer SHALL 识别出耗时超过 YAML 配置中定义的帧预算阈值的帧，并 SHALL 将它们标记为慢帧。
2. 当慢帧被识别出来时，THE Frame_Analyzer SHALL 按各函数在这些慢帧内的聚合耗时对函数排序，并 SHALL 产出一份排序后的热点列表。
3. 当一份热点列表被产出时，THE Analyzer SHALL 概括可能的瓶颈，并 SHALL 引用支撑该概括的具体函数与帧。
4. 如果没有任何帧超过帧预算阈值，那么 THE Frame_Analyzer SHALL 报告在所采集场景中未检测到慢帧。

### 需求 6：通过更细粒度的埋点进行下钻

**用户故事：** 作为一名嵌入式开发者，我希望当粗粒度函数掩盖了真正瓶颈时，智能体能加入更细的 profiler 埋点并重新测试，以便暴露真正的热点。

#### 验收标准

1. 当排序后的热点归结为一个内部耗时分布未知的粗粒度函数时，THE Instrumentor SHALL 在所识别的 C 源码区域中插入 `LV_PROFILER_BEGIN_TAG`/`LV_PROFILER_END_TAG` 成对埋点。
2. THE Instrumentor SHALL 将每个 profiler 标签参数作为字符串字面量输出。
3. 当新的埋点已被插入时，THE DeviceIO SHALL 重新编译固件并烧录目标，且 THE Orchestrator SHALL 针对同一场景重新执行采集与分析。
4. 在已完成的下钻迭代次数低于 Max_Iteration_Limit 期间，THE Orchestrator SHALL 持续下钻，直至隔离出一个叶子级热点或不再存在慢帧。
5. 当达到 Max_Iteration_Limit 时，THE Orchestrator SHALL 停止下钻，并报告迄今为止收集到的最佳热点证据。
6. 在默认时间戳精度比区分下钻标签所需的亚毫秒分辨率更粗的情况下，THE Instrumentor SHALL 报告在更细的计时可被信任之前需要一个微秒级精度的 tick 回调。

### 需求 7：提出并应用优化

**用户故事：** 作为一名嵌入式开发者，我希望智能体能针对已识别的瓶颈提出并应用源码级优化，以便卡顿真正被修复。

#### 验收标准

1. 当一个叶子级热点被隔离出来时，THE Optimizer SHALL 提出一项或多项源码级优化，每一项都引用目标函数以及预期的性能效果。
2. 在配置为 Full_Auto_Mode 的情况下，THE Optimizer SHALL 将所选优化应用到 C 源码，且 THE DeviceIO SHALL 重新编译并烧录目标。
3. 在配置为 Half_Auto_Mode 的情况下，THE Orchestrator SHALL 向开发者呈现所提出的优化，并 SHALL 仅在开发者批准后才应用它。
4. 当一项优化被应用时，THE Orchestrator SHALL 在运行状态中记录所应用的改动及其目标函数。

### 需求 8：验证修复是否改善了帧率

**用户故事：** 作为一名嵌入式开发者，我希望智能体能验证所应用的优化确实改善了帧率，以便我能信任该修复。

#### 验收标准

1. 当一项优化已被应用并烧录时，THE Orchestrator SHALL 重新采集并重新分析优化前所使用的同一 UI 场景。
2. 当优化后的分析完成时，THE Frame_Analyzer SHALL 将优化后的慢帧数量与热点耗时同该场景优化前的基线进行对比。
3. 如果优化后的分析相较基线未显示任何改进，那么 THE Orchestrator SHALL 将该优化标记为无效，并 SHALL 在 Max_Iteration_Limit 之内回退到选择另一项替代优化。
4. 当一项优化被验证为改进时，THE Orchestrator SHALL 将实测效果记录为基线指标与优化后指标之差。

### 需求 9：持久化并复用长期性能知识

**用户故事：** 作为一名嵌入式开发者，我希望智能体记住哪些做法有效，以便它在每次使用后都更聪明、更快。

#### 验收标准

1. 当一项优化被验证为改进时，THE Memory_Store SHALL 持久化一条包含症状、根因、所应用优化与实测效果的 Knowledge_Record。
2. 当一个新的 Tuning_Run 开始时，THE Retriever SHALL 针对 Memory_Store 执行语义召回，并 SHALL 返回按与所报告症状相关性排序的 Knowledge_Record。
3. 在某条被召回的 Knowledge_Record 与当前症状相匹配的情况下，THE Orchestrator SHALL 在发起一次全新下钻之前，将其先前的优化作为候选方案呈现。
4. THE Retriever SHALL 暴露 BaseRetriever 接口，以便可通过 YAML 配置用外部 RAG 后端替换基于本地 Store 的实现。
5. 当一条 Knowledge_Record 被持久化时，THE Memory_Store SHALL 使该记录可跨不同线程供后续的 Tuning_Run 使用。

### 需求 10：可恢复的运行状态

**用户故事：** 作为一名嵌入式开发者，我希望一次调优运行是可恢复的，以便中断不会迫使我重启整个闭环。

#### 验收标准

1. 当 Orchestrator 完成一个 Tuning_Run 的某个阶段时，THE Checkpointer SHALL 在该 Tuning_Run 的 thread id 下持久化运行状态。
2. 当开发者通过 thread id 恢复一个被中断的 Tuning_Run 时，THE Orchestrator SHALL 恢复已持久化的运行状态，并 SHALL 从上一个已完成阶段继续。
3. 在一个 Tuning_Run 于某个 Half_Auto_Mode 干预点暂停期间，THE Checkpointer SHALL 保留运行状态直至开发者恢复该运行。

### 需求 11：全自动与半自动模式及迭代边界

**用户故事：** 作为一名嵌入式开发者，我希望能根据我的环境可脚本化的程度选择全自动或半自动，以便智能体既契合我的配置又不会失控。

#### 验收标准

1. THE Config_Loader SHALL 从 YAML 配置中读取自动化模式与 Max_Iteration_Limit。
2. 在配置为 Full_Auto_Mode 的情况下，THE Orchestrator SHALL 在不为人工输入暂停的前提下运行采集-分析-下钻-优化-验证闭环。
3. 在配置为 Half_Auto_Mode 的情况下，THE Orchestrator SHALL 在每个已配置的干预点暂停，并 SHALL 在继续之前等待开发者确认。
4. 在一个 Tuning_Run 内已完成的闭环迭代次数低于 Max_Iteration_Limit 期间，THE Orchestrator SHALL 被允许开始另一次迭代。
5. 当一个 Tuning_Run 内已完成的闭环迭代次数达到 Max_Iteration_Limit 时，THE Orchestrator SHALL 停止该闭环，并 SHALL 报告已完成工作的摘要。

### 需求 12：以 YAML 驱动的配置及 Factory + Base-Class 提供方

**用户故事：** 作为一名嵌入式开发者，我希望整个系统由 YAML 驱动并建立在一致的工厂抽象之上，以便我无需改动代码即可替换提供方与后端。

#### 验收标准

1. 当系统启动时，THE Config_Loader SHALL 加载 YAML 配置，并 SHALL 校验所选提供方的必填字段均已存在。
2. 如果某个必填配置字段缺失或无效，那么 THE Config_Loader SHALL 在任何智能体运行之前报告具体字段及原因。
3. 当请求一个 LLM_Provider 时，THE LLM_Provider 工厂 SHALL 从 BaseLLMProvider 基类构造 YAML 配置中命名的提供方。
4. 当请求一个 DeviceIO 后端时，THE DeviceIO 工厂 SHALL 从 DeviceIO 基类构造 YAML 配置中命名的后端。
5. 当请求一个 Retriever 后端时，THE Retriever 工厂 SHALL 从 BaseRetriever 基类构造 YAML 配置中命名的后端。

### 需求 13：通过 Subagents-as-Tools 进行 Orchestrator 协调

**用户故事：** 作为一名嵌入式开发者，我希望由一个主智能体将各专职子智能体作为工具来协调，以便推理工作与确定性工作被清晰分离且可靠。

#### 验收标准

1. THE Orchestrator SHALL 使用 LangGraph 的 subagents-as-tools 模式，将 Collector、Analyzer、Instrumentor 与 Optimizer 子智能体暴露为可调用的工具。
2. THE Orchestrator SHALL 将确定性操作——Trace_Filter、Trace_Parser 与 Frame_Analyzer——作为普通工具而非 LLM 子智能体来调用。
3. 当一个子智能体工具返回结果时，THE Orchestrator SHALL 在选择下一个动作之前，将该结果记录到运行状态中。
4. 如果一个子智能体工具返回错误，那么 THE Orchestrator SHALL 记录该错误，并 SHALL 依据所配置的自动化模式决定下一个动作。

### 需求 14：可复用的 DeviceIO 抽象以支持未来自动化

**用户故事：** 作为一名嵌入式开发者，我希望设备控制层是一个可复用的抽象，以便同一框架日后不仅能做性能调优，还能驱动嵌入式功能与自动化测试。

#### 验收标准

1. THE DeviceIO 基类 SHALL 定义 open_serial、capture、build、flash、inject_input 与 send_cmd 各项操作。
2. 在某个具体 DeviceIO 后端无法脚本化 build、flash 或输入注入的情况下，THE 该后端 SHALL 将这些操作暴露为人在环干预点，而非加载失败。
3. THE DeviceIO 抽象 SHALL 可通过同一基类接口被性能调优闭环以外的调用方使用。
4. 当以某个设备命令调用 send_cmd 时，THE DeviceIO SHALL 通过已打开的串口连接发送该命令，并 SHALL 返回设备响应。

### 需求 15：语言无关的 API 边界以支持未来客户端

**用户故事：** 作为一名嵌入式开发者，我希望系统被暴露在一个清晰的、与语言无关的 API 边界之后，以便未来的 TypeScript 客户端或 Web UI 能够在不重写 Python 核心的前提下消费它。

#### 验收标准

1. THE Orchestrator SHALL 通过一个已定义的 API/协议边界（例如 REST 服务层或 MCP 风格的协议）暴露核心能力：提交问题、运行调优、获取分析报告、获取 Knowledge_Record。
2. THE API 契约 SHALL 独立于内部实现语言进行定义，使得外部客户端无需了解核心以 Python 实现即可与之交互。
3. 当一个确定性工件（分析报告、慢帧数据、热点数据）被生成时，THE Orchestrator SHALL 以一种语言中立的格式（例如 JSON）提供该工件，以便外部客户端能够渲染它。
4. THE 慢帧报告与热点数据的语言中立序列化 SHALL 保留渲染火焰图或慢帧报告所需的字段（函数名、时间戳、耗时、线程标识）。
5. 在某个外部客户端通过 API 边界提交问题的情况下，THE Orchestrator SHALL 创建一个 Tuning_Run 并返回其 thread id，使该客户端后续能够获取该运行的状态与工件。
6. 如果通过 API 边界的某个请求引用了不存在的 Tuning_Run 或 Knowledge_Record，那么 THE Orchestrator SHALL 返回一条描述性错误，标明缺失的标识符。

### 需求 16：可插拔能力（Capability）架构

**用户故事：** 作为一名平台开发者，我希望平台的每项功能都是一个遵循统一契约的可插拔能力模块，以便我无需改动核心即可新增或替换能力。

#### 验收标准

1. THE Agent_Platform SHALL 定义一个 BaseCapability 基类，规定每个能力必须提供其名称、面向意图匹配的描述，以及构建自身 subgraph 的方法。
2. 当平台启动时，THE Capability_Factory SHALL 依据注册键构造在 YAML 配置中启用的每个 Capability 实例。
3. 当一个新的 Capability 依据 BaseCapability 契约注册时，THE Agent_Platform SHALL 在不修改 Router_Orchestrator 或 Core_Services 代码的前提下使其可被调用。
4. 每个 Capability SHALL 将自身暴露为一个可被 Router_Orchestrator 调用的工具（subgraph-as-tool）。
5. THE PerfTuning_Capability SHALL 作为 BaseCapability 的一个实现被注册，其内部即需求 1–15 所定义的性能调优闭环。

### 需求 17：Router_Orchestrator 意图分发

**用户故事：** 作为一名嵌入式开发者，我希望向平台描述我的目标，由平台自动选择正确的能力，以便我不必手工指定使用哪个模块。

#### 验收标准

1. 当开发者提交一个请求时，THE Router_Orchestrator SHALL 依据各已注册 Capability 的描述判定意图，并 SHALL 将请求分发到匹配的 Capability。
2. 如果没有任何已注册 Capability 与请求匹配，那么 THE Router_Orchestrator SHALL 返回一条列出可用能力的描述性错误。
3. 如果请求同时匹配多个 Capability，那么 THE Router_Orchestrator SHALL 在分发前请求开发者澄清选择哪一个能力。
4. 当一个 Capability 完成其运行时，THE Router_Orchestrator SHALL 将该 Capability 产出的结果返回给调用方。

### 需求 18：共享底座服务被多能力复用

**用户故事：** 作为一名平台开发者，我希望设备控制、LLM、记忆与配置作为共享底座供所有能力复用，以便各能力不必各自重复实现基础设施。

#### 验收标准

1. THE Agent_Platform SHALL 向每个 Capability 提供一组共享的 Core_Services——DeviceIO、LLM_Provider、Retriever、Memory（Checkpointer + Store）、Config_Loader。
2. THE PerfTuning_Capability 与 THE FunctionalTest_Capability SHALL 通过同一 DeviceIO 基类接口访问设备控制面，不得各自定义独立的设备控制层。
3. 当某个 Capability 持久化或召回知识时，THE Memory SHALL 以能力标识对 Knowledge_Record 分区，使跨能力的知识既可隔离又可按需共享。
4. THE Core_Services SHALL 由平台构造一次并注入各 Capability，而非由各 Capability 自行实例化。

### 需求 19：自动化测试能力（Capability B）契约与骨架

**用户故事：** 作为一名嵌入式开发者，我希望平台预留一个自动化测试能力，复用同一底座，以便日后无需重构平台即可用它驱动嵌入式功能测试。

#### 验收标准

1. THE FunctionalTest_Capability SHALL 作为 BaseCapability 的一个实现存在，并被 Capability_Factory 注册与构造。
2. THE FunctionalTest_Capability SHALL 通过共享的 DeviceIO 接口使用 inject_input、send_cmd 与 capture 操作来驱动设备并观察其响应。
3. THE FunctionalTest_Capability SHALL 定义其测试用例、断言与测试报告的数据契约，使其可被 API 边界以语言中立格式暴露。
4. 在本期 FunctionalTest_Capability 仅交付骨架的情况下，THE 该能力 SHALL 至少提供一个可端到端跑通的最小测试流程（在测试 harness 上运行），以证明平台契约成立，其余业务逻辑作为后续能力实现。

### 需求 20：确定性测试 Harness

**用户故事：** 作为一名平台开发者，我希望有一个可回放的测试 harness，使任意能力都能脱离真实硬件与真实 LLM 运行，以便测试离线、确定且可重复。

#### 验收标准

1. THE Test_Harness SHALL 提供一个 FakeDeviceIO，它实现 DeviceIO 基类接口并回放录制的采集日志与构建/烧录结果，而不触达真实硬件。
2. THE Test_Harness SHALL 提供一个 FakeLLMProvider，它实现 BaseLLMProvider 接口并回放预置的模型决策，而不调用真实 LLM API。
3. 当一个 Capability 在 Test_Harness 上运行时，THE 平台 SHALL 仅通过注入这些 fake 实现即可驱动该能力的完整 subgraph，无需改动能力代码。
4. THE Test_Harness SHALL 使确定性核心工具（Trace_Filter、Trace_Parser、Frame_Analyzer）可用录制 fixtures 独立执行并断言其输出。
