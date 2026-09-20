"""
Benchmark data models and metrics for generative AI performance measurement.

Provides comprehensive data structures for capturing, storing, and analyzing
benchmark results from scheduler-driven generative AI workload executions.
Core abstractions include base benchmark interfaces, generative-specific
metrics with token/latency distributions, request-level statistics tracking,
and multi-benchmark reporting capabilities. These models enable detailed
performance analysis including throughput, latency, concurrency patterns, and
domain-specific metrics for text, image, video, and audio generation tasks.
"""

from __future__ import annotations

import time

from typing import Any, Literal

from pydantic import Field, computed_field

from guidellm.benchmark.schemas.accumulator import (
    GenerativeBenchmarkAccumulator,
)
from guidellm.benchmark.schemas.base import Benchmark, BenchmarkConfig
from guidellm.benchmark.schemas.metrics import (
    GenerativeMetrics,
    SchedulerMetrics,
)
from guidellm.scheduler import SchedulerState
from guidellm.schemas import (
    GenerativeRequestStats,
    StatusBreakdown,
    StatusDistributionSummary,
)

__all__ = ["GenerativeBenchmark"]


class GenerativeBenchmark(Benchmark[GenerativeBenchmarkAccumulator]):
    """Complete generative AI benchmark results with specialized metrics.

    Encapsulates comprehensive performance data from scheduler-driven generative
    workload executions including request-level statistics, token/latency distributions,
    throughput analysis, and concurrency patterns. Provides computed fields for temporal
    analysis and status-grouped request details for detailed post-execution reporting.
    """

    type_: Literal["generative_benchmark"] = "generative_benchmark"  # type: ignore[assignment]

    config: BenchmarkConfig = Field(
        description="Configuration parameters for this benchmark execution",
    )
    scheduler_state: SchedulerState = Field(
        description="Final state of the scheduler after benchmark completion",
    )
    scheduler_metrics: SchedulerMetrics = Field(
        description="Scheduler timing and performance statistics",
    )
    metrics: GenerativeMetrics = Field(
        description="Performance metrics and statistical distributions",
    )
    requests: StatusBreakdown[
        list[GenerativeRequestStats],
        list[GenerativeRequestStats],
        list[GenerativeRequestStats],
        None,
    ] = Field(
        description=(
            "Request details grouped by status: successful, incomplete, errored"
        ),
    )
    quality: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Per-scorer quality aggregates over all scored requests: "
            "{mean, min, max, n} keyed by scorer name. Empty when no "
            "scorers were configured."
        ),
    )
    quality_instrument: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Instrument metadata for the quality scores: scorer names, "
            "formality tier, scope, and compilation timestamp. Scores are "
            "reported as measurement-instrument readings, not bare scalars."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def start_time(self) -> float:
        """
        :return: Benchmark start time in seconds since epoch
        """
        return self.scheduler_metrics.measure_start_time

    @computed_field  # type: ignore[prop-decorator]
    @property
    def end_time(self) -> float:
        """
        :return: Benchmark end time in seconds since epoch
        """
        return self.scheduler_metrics.measure_end_time

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """
        :return: Total benchmark execution duration in seconds
        """
        return self.end_time - self.start_time

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warmup_duration(self) -> float:
        """
        :return: Warmup phase duration in seconds
        """
        return (
            self.scheduler_metrics.measure_start_time
            - self.scheduler_metrics.request_start_time
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cooldown_duration(self) -> float:
        """
        :return: Cooldown phase duration in seconds
        """
        return (
            self.scheduler_metrics.request_end_time
            - self.scheduler_metrics.measure_end_time
        )

    @property
    def request_latency(self) -> StatusDistributionSummary:
        """
        :return: Statistical distribution of request latencies across all requests
        """
        return self.metrics.request_latency

    @property
    def request_throughput(self) -> StatusDistributionSummary:
        """
        :return: Statistical distribution of throughput measured in requests per second
        """
        return self.metrics.requests_per_second

    @property
    def request_concurrency(self) -> StatusDistributionSummary:
        """
        :return: Statistical distribution of concurrent requests throughout execution
        """
        return self.metrics.request_concurrency

    @classmethod
    def compile(
        cls,
        accumulator: GenerativeBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ) -> GenerativeBenchmark:
        """
        Compile final benchmark results from accumulated execution state.

        :param accumulator: Accumulated benchmark state with request statistics
        :param scheduler_state: Final scheduler state after execution completion
        :return: Compiled generative benchmark instance with complete metrics
        """
        quality, quality_instrument = cls._compile_quality(accumulator)
        return GenerativeBenchmark(
            config=accumulator.config,
            scheduler_state=scheduler_state,
            scheduler_metrics=SchedulerMetrics.compile(accumulator, scheduler_state),
            metrics=GenerativeMetrics.compile(accumulator),
            requests=StatusBreakdown(
                successful=accumulator.completed.get_sampled(),
                incomplete=accumulator.incomplete.get_sampled(),
                errored=accumulator.errored.get_sampled(),
                total=None,
            ),
            quality=quality,
            quality_instrument=quality_instrument,
        )

    @staticmethod
    def _compile_quality(
        accumulator: GenerativeBenchmarkAccumulator,
    ) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        """Aggregate per-scorer quality totals into report-ready summaries.

        Returns (quality, quality_instrument). Instrument metadata records
        the scorer names, formality tier, scope and timestamp alongside the
        numbers: scores are instrument readings, not bare scalars.
        """
        quality: dict[str, dict[str, float]] = {}
        for scorer_name, total in accumulator.quality_totals.items():
            n = total["n"]
            quality[scorer_name] = {
                "mean": total["sum"] / n if n else 0.0,
                "min": total["min"] if n else 0.0,
                "max": total["max"] if n else 0.0,
                "n": n,
            }
        instrument = (
            {
                "scorers": sorted(accumulator.quality_totals),
                "formality_tier": "deterministic-instrument",
                "scope": "benchmark completed requests",
                "compiled_at": time.time(),
            }
            if quality
            else {}
        )
        return quality, instrument
