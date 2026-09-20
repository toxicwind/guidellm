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
    # Non-greedy (documented): the first complete span is removed.
    text = "<think>a <think>b</think> c</think> done"
    assert strip_thinking_blocks(text) == "c</think> done"


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

    return GenerativeRequestStats.model_construct(output=output, scores={})


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


def test_score_request_no_output_is_noop():
    acc = _make_accumulator([InstructionFollowingScorer(sentinel=SENTINEL)])
    stats = _make_stats(None)
    acc._score_request(stats)
    assert stats.scores == {}
    assert acc.quality_totals == {}


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
    assert "exploding" not in acc.quality_totals


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
