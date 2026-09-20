"""Pluggable response-quality scoring for benchmarks.

Built-in scorers:
    instruction_following  exact/contains grading borrowed from the estate
                           abstract probe (probe_abstract.py): 2.0 exact,
                           1.0 contains-with-extra, 0.0 miss.

Adapters:
    ThinkingBlockStripper  removes <think>/<reasoning>/<thought>/<scratchpad>
                           spans and fenced thinking blocks before delegating
                           to an inner scorer. Enabled per scorer with
                           ``strip_thinking: true`` in ``scorer_config``.
"""

from __future__ import annotations

from typing import Any

from .adapters import ThinkingBlockStripper, strip_thinking_blocks
from .instruction import InstructionFollowingScorer
from .protocol import Scorer, ScorerResult
from .registry import (
    get_scorer,
    list_scorers,
    register_scorer,
    register_scorer_factory,
)

__all__ = [
    "Scorer",
    "ScorerResult",
    "InstructionFollowingScorer",
    "ThinkingBlockStripper",
    "strip_thinking_blocks",
    "register_scorer",
    "register_scorer_factory",
    "get_scorer",
    "list_scorers",
    "resolve_scorers",
]


def resolve_scorers(
    names: list[str] | None,
    configs: dict[str, dict[str, Any]] | None = None,
) -> list[Scorer]:
    """Build the scorer pipeline for a benchmark run.

    :param names: Scorer names in scoring order.
    :param configs: Per-scorer constructor kwargs, e.g.
        ``{"instruction_following": {"sentinel": "ABSTRACT-7X3Q",
        "strip_thinking": True}}``. The ``strip_thinking`` key is consumed
        here and wraps the scorer in a ThinkingBlockStripper.
    :raises KeyError: on an unknown scorer name.
    """
    configs = configs or {}
    resolved: list[Scorer] = []
    for name in names or []:
        cfg = dict(configs.get(name, {}))
        strip_thinking = cfg.pop("strip_thinking", False)
        scorer = get_scorer(name, **cfg)
        if strip_thinking:
            scorer = ThinkingBlockStripper(scorer)
        resolved.append(scorer)
    return resolved
