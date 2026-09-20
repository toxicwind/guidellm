"""Scorer protocol for pluggable response-quality scoring.

A scorer grades a single generated response and returns a numeric score plus
diagnostic details. Scorers are measurement instruments: the report records
the scorer name alongside every number (cf. the "judge datasheet" argument
that a judge should be reported as an instrument, not just a scalar).

Design note: deterministic instruction-following scorers are the primary
quality signal. LLM-as-judge carries position, verbosity and self-enhancement
biases (Zheng et al., 2306.05685) and rubric artifacts (2609.02942); a judge
may be added later as a second scorer, never as the only one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["ScorerResult", "Scorer"]


@dataclass
class ScorerResult:
    """Outcome of scoring one response."""

    score: float
    """Numeric score. Convention is [0, 2] unless the scorer documents otherwise."""

    name: str
    """Name of the scorer that produced this result."""

    details: dict[str, Any] = field(default_factory=dict)
    """Diagnostics, e.g. match kind, chars stripped, error info."""


@runtime_checkable
class Scorer(Protocol):
    """Structural interface every scorer implements."""

    name: str
    """Registry name of this scorer."""

    def score(
        self,
        output: str,
        expected: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ScorerResult:
        """Score one generated response.

        :param output: The model-generated response text.
        :param expected: Optional expected answer; overrides scorer defaults.
        :param context: Optional extra inputs (prompt, request id, ...).
        :return: The score plus diagnostics. Must not raise on bad input.
        """
        ...
