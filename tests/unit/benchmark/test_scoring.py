"""Unit tests for pluggable response-quality scoring."""

import pytest

from guidellm.benchmark.scoring import (
    InstructionFollowingScorer,
    ScorerResult,
    ThinkingBlockStripper,
    get_scorer,
    list_scorers,
    register_scorer,
    resolve_scorers,
    strip_thinking_blocks,
)

SENTINEL = "ABSTRACT-7X3Q"


# ---------------------------------------------------------------------------
# InstructionFollowingScorer: semantics borrowed from probe_abstract.py
# ---------------------------------------------------------------------------


def test_instruction_exact_match_scores_2():
    scorer = InstructionFollowingScorer(sentinel=SENTINEL)
    result = scorer.score(SENTINEL)
    assert result.score == 2.0
    assert result.details["match"] == "exact"


def test_instruction_exact_match_ignores_surrounding_whitespace():
    scorer = InstructionFollowingScorer(sentinel=SENTINEL)
    result = scorer.score("  \n" + SENTINEL + "\t\n")
    assert result.score == 2.0
    assert result.details["match"] == "exact"


def test_instruction_contains_with_extra_scores_1():
    scorer = InstructionFollowingScorer(sentinel=SENTINEL)
    result = scorer.score("The code is " + SENTINEL + " as requested.")
    assert result.score == 1.0
    assert result.details["match"] == "contains"


def test_instruction_miss_scores_0():
    scorer = InstructionFollowingScorer(sentinel=SENTINEL)
    result = scorer.score("something entirely different")
    assert result.score == 0.0
    assert result.details["match"] == "none"


def test_instruction_empty_output_scores_0():
    scorer = InstructionFollowingScorer(sentinel=SENTINEL)
    assert scorer.score("").score == 0.0
    assert scorer.score(None).score == 0.0


def test_instruction_no_sentinel_scores_0():
    scorer = InstructionFollowingScorer()
    result = scorer.score(SENTINEL)
    assert result.score == 0.0
    assert result.details["reason"] == "no-sentinel"


def test_instruction_expected_overrides_constructor_sentinel():
    scorer = InstructionFollowingScorer(sentinel="WRONG")
    assert scorer.score(SENTINEL, expected=SENTINEL).score == 2.0
    assert scorer.score(SENTINEL).score == 0.0


def test_instruction_casefold_option():
    scorer = InstructionFollowingScorer(sentinel=SENTINEL, casefold=True)
    assert scorer.score(SENTINEL.lower()).score == 2.0
    strict = InstructionFollowingScorer(sentinel=SENTINEL)
    assert strict.score(SENTINEL.lower()).score == 0.0


def test_instruction_result_carries_scorer_name():
    scorer = InstructionFollowingScorer(sentinel=SENTINEL)
    result = scorer.score(SENTINEL)
    assert isinstance(result, ScorerResult)
    assert result.name == "instruction_following"


# ---------------------------------------------------------------------------
# strip_thinking_blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tag", ["think", "reasoning", "thought", "scratchpad"])
def test_strip_thinking_blocks_each_tag(tag):
    text = f"before <{tag}>internal monologue</{tag}> after"
    assert strip_thinking_blocks(text) == "before  after"


def test_strip_thinking_blocks_case_insensitive():
    text = "A <THINK>deep thought</ThInK> B"
    assert strip_thinking_blocks(text) == "A  B"


def test_strip_thinking_blocks_non_greedy():
    text = "<think>one</think> keep <think>two</think>"
    assert strip_thinking_blocks(text) == "keep"


def test_strip_thinking_blocks_nested_same_tag():
    # True nesting: innermost spans collapse first, the whole nested
    # block is thinking content.
    text = "<think>a <think>b</think> c</think> done"
    assert strip_thinking_blocks(text) == "done"


def test_strip_thinking_blocks_deeply_nested():
    text = "<think>a <think>b <think>c</think> d</think> e</think> X"
    assert strip_thinking_blocks(text) == "X"


def test_strip_thinking_blocks_orphan_closing_tag_removed():
    # Malformed residue: a lone closing tag is not answer content.
    assert strip_thinking_blocks("c</think> done") == "c done"


def test_strip_thinking_blocks_nested_keeps_between_blocks():
    text = "<think>one</think> keep <think>two <think>nested</think></think>"
    assert strip_thinking_blocks(text) == "keep"


def test_strip_thinking_blocks_unclosed_tag_strips_to_end():
    text = "answer here <reasoning>never closed..."
    assert strip_thinking_blocks(text) == "answer here"


def test_strip_thinking_blocks_fenced():
    text = "```thinking\nchain of thought\n```\n" + SENTINEL
    assert strip_thinking_blocks(text) == SENTINEL


def test_strip_thinking_blocks_passthrough_without_blocks():
    text = "plain answer, no blocks"
    assert strip_thinking_blocks(text) == text


def test_strip_thinking_blocks_empty():
    assert strip_thinking_blocks("") == ""


# ---------------------------------------------------------------------------
# ThinkingBlockStripper adapter
# ---------------------------------------------------------------------------


def test_adapter_strips_before_delegating():
    inner = InstructionFollowingScorer(sentinel=SENTINEL)
    scorer = ThinkingBlockStripper(inner)
    result = scorer.score("<think>deliberation</think>" + SENTINEL)
    assert result.score == 2.0
    assert result.name == "instruction_following_nothink"
    assert result.details["stripped"] is True
    assert result.details["chars_removed"] > 0


def test_adapter_reports_no_strip_when_clean():
    inner = InstructionFollowingScorer(sentinel=SENTINEL)
    scorer = ThinkingBlockStripper(inner)
    result = scorer.score(SENTINEL)
    assert result.score == 2.0
    assert result.details["stripped"] is False
    assert result.details["chars_removed"] == 0


def test_adapter_composes_inner_details():
    inner = InstructionFollowingScorer(sentinel=SENTINEL)
    scorer = ThinkingBlockStripper(inner)
    result = scorer.score("<think>x</think>extra " + SENTINEL)
    assert result.score == 1.0
    assert result.details["match"] == "contains"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_builtin_scorer_registered():
    assert "instruction_following" in list_scorers()


def test_registry_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        get_scorer("does_not_exist")


def test_registry_instance_roundtrip():
    scorer = InstructionFollowingScorer(sentinel=SENTINEL)
    register_scorer("test_instance_scorer", scorer)
    assert get_scorer("test_instance_scorer") is scorer
    assert "test_instance_scorer" in list_scorers()


def test_registry_kwargs_on_instance_raises():
    register_scorer("test_kwarg_guard", InstructionFollowingScorer(sentinel=SENTINEL))
    with pytest.raises(ValueError):
        get_scorer("test_kwarg_guard", sentinel="X")


def test_registry_factory_builds_with_kwargs():
    scorer = get_scorer("instruction_following", sentinel=SENTINEL)
    assert isinstance(scorer, InstructionFollowingScorer)
    assert scorer.score(SENTINEL).score == 2.0


# ---------------------------------------------------------------------------
# resolve_scorers
# ---------------------------------------------------------------------------


def test_resolve_scorers_none_and_empty():
    assert resolve_scorers(None) == []
    assert resolve_scorers([]) == []


def test_resolve_scorers_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        resolve_scorers(["nope"])


def test_resolve_scorers_passes_config_and_wraps_strip_thinking():
    scorers = resolve_scorers(
        ["instruction_following"],
        {"instruction_following": {"sentinel": SENTINEL, "strip_thinking": True}},
    )
    assert len(scorers) == 1
    scorer = scorers[0]
    assert isinstance(scorer, ThinkingBlockStripper)
    assert isinstance(scorer.inner, InstructionFollowingScorer)
    assert scorer.score("<think>x</think>" + SENTINEL).score == 2.0


def test_resolve_scorers_without_strip_thinking_flag():
    (scorer,) = resolve_scorers(
        ["instruction_following"],
        {"instruction_following": {"sentinel": SENTINEL}},
    )
    assert isinstance(scorer, InstructionFollowingScorer)


# ---------------------------------------------------------------------------
# Accumulator integration: _score_request + quality_totals
# ---------------------------------------------------------------------------


def _make_accumulator(scorers):
    from types import SimpleNamespace

    from guidellm.benchmark.schemas.accumulator import GenerativeBenchmarkAccumulator

    # model_construct runs model_post_init, which needs these config attrs.
    config = SimpleNamespace(sample_size=None, scorers=[], scorer_config={})
    acc = GenerativeBenchmarkAccumulator.model_construct(config=config)
    # Override the (empty) resolved pipeline with the test's scorers.
    acc._scorers = scorers
    acc.quality_totals = {}
    return acc


def _make_stats(output):
    from guidellm.schemas.base.request_stats import GenerativeRequestStats

    return GenerativeRequestStats.model_construct(
        output=output, scores={}, score_details={}
    )


def test_score_request_records_scores_and_totals():
    acc = _make_accumulator(
        [InstructionFollowingScorer(sentinel=SENTINEL)]
    )
    stats = _make_stats(SENTINEL)
    acc._score_request(stats)
    assert stats.scores == {"instruction_following": 2.0}
    total = acc.quality_totals["instruction_following"]
    assert total == {"sum": 2.0, "min": 2.0, "max": 2.0, "n": 1.0}


def test_score_request_accumulates_min_max():
    acc = _make_accumulator([InstructionFollowingScorer(sentinel=SENTINEL)])
    for output in (SENTINEL, "extra " + SENTINEL, "miss"):
        acc._score_request(_make_stats(output))
    total = acc.quality_totals["instruction_following"]
    assert total["n"] == 3.0
    assert total["sum"] == pytest.approx(3.0)
    assert total["min"] == 0.0
    assert total["max"] == 2.0


def test_score_request_empty_output_scores_zero():
    acc = _make_accumulator([InstructionFollowingScorer(sentinel=SENTINEL)])
    stats = _make_stats("")
    acc._score_request(stats)
    assert stats.scores == {"instruction_following": 0.0}
    assert stats.score_details["instruction_following"]["match"] == "none"
    total = acc.quality_totals["instruction_following"]
    assert total["n"] == 1.0 and total["sum"] == 0.0


def test_score_request_none_output_scores_zero():
    acc = _make_accumulator([InstructionFollowingScorer(sentinel=SENTINEL)])
    stats = _make_stats(None)
    acc._score_request(stats)
    assert stats.scores == {"instruction_following": 0.0}
    assert acc.quality_totals["instruction_following"]["n"] == 1.0


def test_score_request_no_scorers_is_noop():
    acc = _make_accumulator([])
    stats = _make_stats(SENTINEL)
    acc._score_request(stats)
    assert stats.scores == {}
    assert acc.quality_totals == {}


def test_score_request_failure_records_zero_not_raise():
    class Exploding:
        name = "exploding"

        def score(self, output, expected=None, context=None):
            raise RuntimeError("boom")

    acc = _make_accumulator([Exploding()])
    stats = _make_stats(SENTINEL)
    acc._score_request(stats)  # must not raise
    assert stats.scores == {"exploding": 0.0}
    # the zero contributes to aggregates ...
    total = acc.quality_totals["exploding"]
    assert total == {"sum": 0.0, "min": 0.0, "max": 0.0, "n": 1.0}
    # ... and the error is persisted in the request's details
    assert "boom" in stats.score_details["exploding"]["error"]


def test_score_request_aggregate_false_records_scores_not_totals():
    """Errored requests: scored for the record, excluded from aggregates."""
    acc = _make_accumulator([InstructionFollowingScorer(sentinel="ABSTRACT-7X3Q")])
    stats = _make_stats("ABSTRACT-7X3Q")
    acc._score_request(stats, aggregate=False)
    assert stats.scores == {"instruction_following": 2.0}
    assert stats.score_details["instruction_following"]["match"] == "exact"
    assert acc.quality_totals == {}


def test_score_request_aggregate_false_exception_still_zero_no_total():
    class Exploding:
        name = "exploding"

        def score(self, output, expected=None, context=None):
            raise RuntimeError("boom")

    acc = _make_accumulator([Exploding()])
    stats = _make_stats("whatever")
    acc._score_request(stats, aggregate=False)
    assert stats.scores == {"exploding": 0.0}
    assert "RuntimeError" in stats.score_details["exploding"]["error"]
    assert acc.quality_totals == {}


def test_score_request_aggregate_true_default_unchanged():
    acc = _make_accumulator([InstructionFollowingScorer(sentinel="ABSTRACT-7X3Q")])
    stats = _make_stats("ABSTRACT-7X3Q")
    acc._score_request(stats)  # default aggregate=True
    assert acc.quality_totals["instruction_following"]["n"] == 1.0


def test_score_request_persists_details():
    acc = _make_accumulator(
        [ThinkingBlockStripper(InstructionFollowingScorer(sentinel=SENTINEL))]
    )
    stats = _make_stats("<think>deliberation</think>" + SENTINEL)
    acc._score_request(stats)
    details = stats.score_details["instruction_following_nothink"]
    assert details["match"] == "exact"
    assert details["stripped"] is True
    assert details["chars_removed"] > 0


def test_no_scorer_request_shape_empty():
    # No scorers configured: scores and details stay empty, request shape
    # is untouched.
    acc = _make_accumulator([])
    stats = _make_stats(SENTINEL)
    acc._score_request(stats)
    assert stats.scores == {}
    assert stats.score_details == {}
    assert acc.quality_totals == {}


# ---------------------------------------------------------------------------
# compile(): quality aggregation + instrument metadata
# ---------------------------------------------------------------------------


def test_compile_quality_aggregates():
    from guidellm.benchmark.schemas.benchmark import GenerativeBenchmark

    acc = _make_accumulator([])
    acc.quality_totals = {
        "instruction_following": {"sum": 3.0, "min": 0.0, "max": 2.0, "n": 2.0}
    }
    quality, instrument = GenerativeBenchmark._compile_quality(acc)
    assert quality == {
        "instruction_following": {"mean": 1.5, "min": 0.0, "max": 2.0, "n": 2.0}
    }
    assert instrument["scorers"] == ["instruction_following"]
    assert instrument["formality_tier"] == "deterministic-instrument"
    assert instrument["scope"] == "benchmark completed requests"
    assert instrument["compiled_at"] > 0


def test_compile_quality_empty_when_no_scorers():
    from guidellm.benchmark.schemas.benchmark import GenerativeBenchmark

    acc = _make_accumulator([])
    acc.quality_totals = {}
    quality, instrument = GenerativeBenchmark._compile_quality(acc)
    assert quality == {}
    assert instrument == {}


# ---------------------------------------------------------------------------
# Config schema: scorer fields present with safe defaults
# ---------------------------------------------------------------------------


def test_benchmark_config_scorer_fields():
    from guidellm.benchmark.schemas.base import BenchmarkConfig

    config = BenchmarkConfig.model_construct()
    assert config.scorers == []
    assert config.scorer_config == {}


def test_metrics_args_scorer_fields():
    from guidellm.schemas.benchmark.entrypoints import GenerativeMetricsArgs

    args = GenerativeMetricsArgs.model_construct(
        scorers=["instruction_following"],
        scorer_config={"instruction_following": {"sentinel": SENTINEL}},
    )
    assert args.scorers == ["instruction_following"]
    assert args.scorer_config["instruction_following"]["sentinel"] == SENTINEL


def test_metrics_args_scorer_defaults_empty():
    from guidellm.schemas.benchmark.entrypoints import GenerativeMetricsArgs

    args = GenerativeMetricsArgs.model_construct()
    assert args.scorers == []
    assert args.scorer_config == {}


# ---------------------------------------------------------------------------
# No-scorer report shape regression
#
# A benchmark compiled with NO scorers configured must serialize to the same
# JSON shape as the pre-scoring schema (commit 74ec8623^ of toxicwind/guidellm).
# The scoring feature is additive-only: nothing may be removed or retyped, and
# the new scoring fields must serialize empty on a no-scorer run.
#
# Pre-scoring field lists below were read from
# `git show 74ec8623^:<path>`:
#   - src/guidellm/benchmark/schemas/benchmark.py  (GenerativeBenchmark)
#   - src/guidellm/schemas/base/request_stats.py   (GenerativeRequestStats)
#   - src/guidellm/benchmark/schemas/base.py       (BenchmarkConfig)
# ---------------------------------------------------------------------------

from guidellm.benchmark.schemas import (
    BenchmarkConfig,
    GenerativeBenchmark,
    GenerativeBenchmarkAccumulator,
)
from guidellm.scheduler import ConcurrentStrategy, SchedulerState
from guidellm.schemas import (
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
    UsageMetrics,
)

_BASE_TIME = 1000.0  # non-zero epoch base; a window starting at 0.0 reads unset


def _json_type(value) -> str:
    """Coarse JSON type of a model_dump(mode="json") value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise AssertionError(f"non-JSON value of type {type(value)}: {value!r}")


# Pre-scoring (74ec8623^) top-level keys of GenerativeBenchmark -> JSON types.
_PRE_SCORING_BENCHMARK_TYPES = {
    "type_": {"string"},
    "config": {"object"},
    "scheduler_state": {"object"},
    "scheduler_metrics": {"object"},
    "metrics": {"object"},
    "requests": {"object"},
    "start_time": {"number"},
    "end_time": {"number"},
    "duration": {"number"},
    "warmup_duration": {"number"},
    "cooldown_duration": {"number"},
}

# Pre-scoring (74ec8623^) keys of GenerativeRequestStats -> JSON types.
_PRE_SCORING_REQUEST_TYPES = {
    "type_": {"string"},
    "request_id": {"string"},
    "response_id": {"string", "null"},
    "request_args": {"string", "null"},
    "output": {"string", "null"},
    "reasoning_output": {"string", "null"},
    "tool_calls": {"array", "null"},
    "info": {"object"},
    "input_metrics": {"object"},
    "output_metrics": {"object"},
    "request_start_time": {"number", "null"},
    "request_end_time": {"number"},
    "request_latency": {"number", "null"},
    "request_dispatch_delay": {"number", "null"},
    "request_scheduled_latency": {"number", "null"},
    "prompt_tokens": {"integer", "null"},
    "cached_tokens": {"integer", "null"},
    "output_tokens": {"integer", "null"},
    "total_tokens": {"integer", "null"},
    "time_to_first_token_ms": {"number", "null"},
    "time_to_last_round_trip_ms": {"number", "null"},
    "avg_round_trip_time_ms": {"number", "null"},
    "time_per_output_token_ms": {"number", "null"},
    "inter_token_latency_ms": {"number", "null"},
    "tokens_per_second": {"number", "null"},
    "output_tokens_per_second": {"number", "null"},
    "time_to_first_output_token_ms": {"number", "null"},
    "iter_tokens_per_iteration": {"number", "null"},
    "output_tokens_per_iteration": {"number", "null"},
}

# Pre-scoring (74ec8623^) keys of BenchmarkConfig (serialized `config` block).
_PRE_SCORING_CONFIG_KEYS = {
    "backend",
    "constraints",
    "cooldown",
    "environment",
    "id_",
    "prefer_response_metrics",
    "profile",
    "requests",
    "run_id",
    "run_index",
    "sample_size",
    "slo",
    "strategy",
    "warmup",
}

# The only additive fields the scoring feature may introduce (all empty when
# no scorers are configured).
# With conditional omission, a no-scorer run serializes with ZERO schema
# deltas vs the pre-scoring schema: scoring-only fields are omitted when empty.
_ADDITIVE_TOP_LEVEL: set = set()
_ADDITIVE_REQUEST: set = set()
_ADDITIVE_CONFIG: set = set()


def _compile_no_scorer_benchmark() -> GenerativeBenchmark:
    """Compile a real benchmark through the accumulator with scoring disabled.

    No live backend: one synthetic completed request, no `scorers` passed to
    BenchmarkConfig so the scoring defaults (empty) apply end to end.
    """
    timings = RequestTimings(
        resolve_start=_BASE_TIME,
        resolve_end=_BASE_TIME + 0.5,
        request_start=_BASE_TIME,
        request_end=_BASE_TIME + 0.5,
    )
    stats = GenerativeRequestStats(
        request_id="req-shape-1",
        request_args="--prompt hello",
        output="hello world",
        info=RequestInfo(
            request_id="req-shape-1", status="completed", timings=timings
        ),
        input_metrics=UsageMetrics(text_tokens=8),
        output_metrics=UsageMetrics(text_tokens=16),
    )
    accumulator = GenerativeBenchmarkAccumulator(
        config=BenchmarkConfig(
            run_id="shape-regression",
            run_index=0,
            strategy=ConcurrentStrategy(streams=1),
            constraints={},
            profile={},
            requests={},
            backend={},
            environment={},
            # scorers / scorer_config intentionally omitted -> disabled
        )
    )
    accumulator.timings.measure_start = _BASE_TIME
    accumulator.timings.measure_end = _BASE_TIME + 10.0
    accumulator.completed.requests_stats = [stats]
    return GenerativeBenchmark.compile(
        accumulator=accumulator, scheduler_state=SchedulerState()
    )


def _assert_shape_matches(payload: dict, pre_scoring_types: dict, additive: set):
    pre_keys = set(pre_scoring_types)
    actual_keys = set(payload)
    removed = sorted(pre_keys - actual_keys)
    assert not removed, f"pre-scoring keys removed by scoring feature: {removed}"
    added = actual_keys - pre_keys
    assert added == additive, (
        f"unexpected schema deltas vs pre-scoring (74ec8623^): {sorted(added)}; "
        f"allowed additive fields: {sorted(additive)}"
    )
    mistyped = {
        key: _json_type(payload[key])
        for key in sorted(pre_keys)
        if _json_type(payload[key]) not in pre_scoring_types[key]
    }
    assert not mistyped, (
        "pre-scoring keys retyped by scoring feature "
        f"(key -> observed JSON type): {mistyped}"
    )


def test_no_scorer_benchmark_top_level_shape_matches_pre_scoring_schema():
    """Top-level JSON shape of a no-scorer run == pre-scoring schema EXACTLY:
    empty scoring fields are omitted, not serialized empty."""
    report = _compile_no_scorer_benchmark().model_dump(mode="json")
    _assert_shape_matches(report, _PRE_SCORING_BENCHMARK_TYPES, _ADDITIVE_TOP_LEVEL)
    assert "quality" not in report
    assert "quality_instrument" not in report


def test_no_scorer_request_shape_matches_pre_scoring_schema():
    """Per-request JSON shape of a no-scorer run == pre-scoring schema EXACTLY:
    empty scoring fields are omitted, not serialized empty."""
    benchmark = _compile_no_scorer_benchmark()
    successful = benchmark.requests.successful
    assert len(successful) >= 1, "need at least one request to check the shape"
    for stats in successful:
        payload = stats.model_dump(mode="json")
        _assert_shape_matches(payload, _PRE_SCORING_REQUEST_TYPES, _ADDITIVE_REQUEST)
        assert "scores" not in payload
        assert "score_details" not in payload


def test_empty_string_sentinel_scores_zero():
    # Scout finding 1: sentinel="" must behave as "no sentinel configured"
    # (0.0), never as a perfect 2.0 on whitespace-only output.
    scorer = InstructionFollowingScorer(sentinel="")
    assert scorer.score("   ").score == 0.0
    assert scorer.score("   ").details["reason"] == "no-sentinel"
    assert scorer.score("anything").score == 0.0


def test_miss_path_details_carry_reason():
    # Scout finding 8: the plain-miss path sets details["reason"] like the
    # other "none" paths, so consumers can rely on the key.
    scorer = InstructionFollowingScorer(sentinel=SENTINEL)
    details = scorer.score("unrelated text").details
    assert details["match"] == "none"
    assert details["reason"] == "no-match"


def test_scorer_name_keying_consistent_on_exception_and_success():
    # Scout finding 2: scores/details/totals key by scorer.name on both the
    # success path and the exception path (was result.name on success).
    class Renamer:
        name = "renamer"

        def __init__(self, fail=False):
            self.fail = fail

        def score(self, output, expected=None, context=None):
            if self.fail:
                raise RuntimeError("kaput")
            return ScorerResult(score=1.0, name="renamed", details={})

    acc = _make_accumulator([Renamer(fail=False)])
    stats = _make_stats(SENTINEL)
    acc._score_request(stats)
    assert set(stats.scores) == {"renamer"}
    assert set(stats.score_details) == {"renamer"}
    assert set(acc.quality_totals) == {"renamer"}

    acc2 = _make_accumulator([Renamer(fail=True)])
    stats2 = _make_stats(SENTINEL)
    acc2._score_request(stats2)
    assert set(stats2.scores) == {"renamer"}
    assert set(stats2.score_details) == {"renamer"}
    assert set(acc2.quality_totals) == {"renamer"}


def test_chars_removed_ignores_whitespace_normalization():
    # Scout finding 3: chars_removed measures block markup only; a padded
    # answer with no blocks reports 0 even though ends are trimmed.
    inner = InstructionFollowingScorer(sentinel=SENTINEL)
    scorer = ThinkingBlockStripper(inner)
    result = scorer.score("  " + SENTINEL + "  ")
    assert result.details["stripped"] is False
    assert result.details["chars_removed"] == 0
    assert result.score == 2.0


def test_no_scorer_config_additive_scoring_keys_only():
    """The serialized `config` block matches the pre-scoring config keys
    EXACTLY: empty scoring config keys are omitted, not serialized empty."""
    report = _compile_no_scorer_benchmark().model_dump(mode="json")
    config = report["config"]
    assert _PRE_SCORING_CONFIG_KEYS <= set(config), (
        "pre-scoring config keys removed: "
        f"{sorted(_PRE_SCORING_CONFIG_KEYS - set(config))}"
    )
    added = set(config) - _PRE_SCORING_CONFIG_KEYS
    assert added == _ADDITIVE_CONFIG, (
        f"unexpected config deltas vs pre-scoring (74ec8623^): {sorted(added)}"
    )
    assert "scorers" not in config
    assert "scorer_config" not in config
