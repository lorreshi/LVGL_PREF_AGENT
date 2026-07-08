"""任务 9.1：日志处理链集成测试（Tier 2）。

用一份真实样例 LVGL profiler 日志端到端串联三个**确定性普通工具**（需求 13.2）：

    原始样例日志 → Trace_Filter → Trace_Parser → Frame_Analyzer

并断言最终 ``FrameReport``（慢帧、热点排名、聚合统计）与该 fixture 的**已知答案**
逐字段一致。样例日志置于 ``tests/fixtures/lvgl_profiler_sample.log``，含：

* 4 个 ``lv_timer_handler`` 顶层调用（帧），其中 **2 个超帧预算**（慢帧）；
* 每帧内 ``lv_draw_rect`` / ``lv_refr_area`` 等热点子调用；
* 2 行**交错损坏**噪声（其他线程插入 / 串口误码），验证 Trace_Filter 的排除与
  受损区间报告不影响下游已知答案。

覆盖的需求（端到端）：

* **需求 3.1**：Trace_Filter 把原始日志转换为已清洗 systrace 工件。
* **需求 4.1**：Trace_Parser 把事件解析为按线程调用树。
* **需求 5.1**：Frame_Analyzer 识别超帧预算的慢帧。

不触达真实硬件 / LLM / 网络：输入为磁盘上的静态 fixture，已清洗工件写入 ``tmp_path``
（不污染仓库 fixtures 目录）。

**Validates: Requirements 3.1, 4.1, 5.1**
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from embedded_device_agent.capabilities.perf_tuning.tools.frame_analyzer import (
    analyze_frames,
)
from embedded_device_agent.capabilities.perf_tuning.tools.systrace_parser import (
    parse_systrace,
)
from embedded_device_agent.capabilities.perf_tuning.tools.trace_filter import (
    trace_filter,
)
from embedded_device_agent.core.config.models import (
    AppConfig,
    DeviceConfig,
    LLMConfig,
    RetrieverConfig,
)
from embedded_device_agent.core.models import RawTraceArtifact

# 仓库自带的真实样例 LVGL profiler 日志 fixture。
_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lvgl_profiler_sample.log"

# 目标 60fps → 帧预算 16667us；样例中 30000us / 25000us 两帧超预算。
_FRAME_BUDGET_US = 16667


def _make_config() -> AppConfig:
    """构造一份最小合法 AppConfig（不触达真实后端），提供分析所需预算参数。"""
    return AppConfig(
        llm=LLMConfig(type="anthropic", model="claude", api_key_env="ANTHROPIC_API_KEY"),
        device=DeviceConfig(type="fake"),
        retriever=RetrieverConfig(),
        frame_budget_us=_FRAME_BUDGET_US,
        hotspot_top_n=20,
        report_token_budget=0,  # 不限制，断言完整已知答案
        mode="full_auto",
        max_iterations=5,
    )


def _raw_artifact() -> RawTraceArtifact:
    """把 fixture 包装为一次采集产出的 RawTraceArtifact（其余字段取合理占位值）。"""
    return RawTraceArtifact(
        run_id="run-integration",
        path=_FIXTURE,
        captured_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        duration_s=1.0,
        baud=921600,
    )


def test_log_pipeline_end_to_end_matches_known_answer(tmp_path: Path) -> None:
    """原始样例日志 → Trace_Filter → Trace_Parser → Frame_Analyzer 的已知答案。"""
    assert _FIXTURE.exists(), f"样例日志缺失：{_FIXTURE}"

    cfg = _make_config()
    clean_path = tmp_path / "clean.systrace"

    # ---- 阶段 1：Trace_Filter（需求 3.1）----------------------------------
    filter_result = trace_filter(_raw_artifact(), clean_path=clean_path)

    # 20 个成对事件全部保留；2 行交错噪声被排除并报告为受损区间（第 13、18 行）。
    assert filter_result.retained == 20
    assert filter_result.excluded == 0
    assert filter_result.corrupted_regions == [(13, 13), (18, 18)]
    assert filter_result.clean_path == clean_path
    assert clean_path.exists()

    # ---- 阶段 2：Trace_Parser（需求 4.1）----------------------------------
    parse_result = parse_systrace(filter_result)

    # 单线程（tid=1）4 帧根调用，全部成对 → 无未匹配事件。
    assert set(parse_result.trees_by_tid.keys()) == {1}
    assert len(parse_result.trees_by_tid[1]) == 4
    assert parse_result.unmatched == []

    # ---- 阶段 3：Frame_Analyzer（需求 5.1）--------------------------------
    report = analyze_frames(parse_result, cfg, scenario="scroll")

    # 聚合统计：4 帧，耗时 [8000, 30000, 12000, 25000] 的 P95（最近秩）= 30000。
    assert report.no_slow_frames is False
    assert report.total_frames == 4
    assert report.p95_frame_us == 30_000
    assert report.scenario == "scroll"
    assert report.frame_budget_us == _FRAME_BUDGET_US
    assert report.target_fps == 60
    assert report.summary is None  # 确定性工具留空，待 LLM Analyzer 填写

    # 慢帧（需求 5.1）：帧序按 begin 排序 → 索引 1（30000us）与 3（25000us）超预算。
    assert [sf.index for sf in report.slow_frames] == [1, 3]
    assert [sf.duration_us for sf in report.slow_frames] == [30_000, 25_000]
    assert [sf.dominant_func for sf in report.slow_frames] == [
        "lv_draw_rect",
        "lv_draw_rect",
    ]

    # 热点排名（需求 5.2）：按跨慢帧聚合自耗时降序。
    #   lv_draw_rect : 20000 + 18000 = 38000（count 2）
    #   lv_refr_area :  5000 +  4000 =  9000（count 2）
    #   lv_timer_handler: 5000 + 3000 = 8000（count 2，root 自耗时 = 帧耗时 - 子耗时）
    assert [h.func for h in report.hotspots] == [
        "lv_draw_rect",
        "lv_refr_area",
        "lv_timer_handler",
    ]
    assert [h.total_us for h in report.hotspots] == [38_000, 9_000, 8_000]
    assert [h.call_count for h in report.hotspots] == [2, 2, 2]
    assert [h.rank for h in report.hotspots] == [1, 2, 3]
