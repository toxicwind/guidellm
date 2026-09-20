"""Instruction-following scorer.

Borrowed semantics from the estate abstract probe
(projects/openrouter-probe/probe_abstract.py): the prompt demands an
exact sentinel-only response, with zero domain content.

Scoring (identical to the probe):
    2.0  normalized output exactly equals the sentinel
    1.0  sentinel appears in the output with extra content around it
    0.0  sentinel missing, empty output, or no sentinel configured

The sentinel is a constructor parameter, never a hardcoded secret: the
benchmark run injects it via ``scorer_config``.
"""

from __future__ import annotations

from typing import Any

from .protocol import ScorerResult
from .registry import register_scorer_factory

__all__ = ["InstructionFollowingScorer"]

EXACT = 2.0
CONTAINS = 1.0
MISS = 0.0


class InstructionFollowingScorer:
    """Grade whether a response follows an exact-output instruction."""

    name = "instruction_following"

    def __init__(
        self,
        sentinel: str | None = None,
        strip: bool = True,
        casefold: bool = False,
    ) -> None:
        """Create the scorer.

        :param sentinel: The exact token the prompt demanded. None disables
            matching (every output scores 0.0).
        :param strip: Strip surrounding whitespace before comparing (probe parity).
        :param casefold: Case-insensitive comparison. Off by default to match
            the probe exactly.
        """
        self.sentinel = sentinel
        self.strip = strip
        self.casefold = casefold

    def _normalize(self, text: str) -> str:
        if self.strip:
            text = text.strip()
        if self.casefold:
            text = text.casefold()
        return text

    def score(
        self,
        output: str,
        expected: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ScorerResult:
        sentinel = expected if expected is not None else self.sentinel
        details: dict[str, Any] = {"sentinel": sentinel}
        if not output or not sentinel:
            details["match"] = "none"
            details["reason"] = "no-sentinel" if not sentinel else "empty-output"
            return ScorerResult(score=MISS, name=self.name, details=details)
        norm_output = self._normalize(output)
        norm_sentinel = self._normalize(sentinel)
        if norm_output == norm_sentinel:
            details["match"] = "exact"
            return ScorerResult(score=EXACT, name=self.name, details=details)
        if norm_sentinel and norm_sentinel in norm_output:
            details["match"] = "contains"
            return ScorerResult(score=CONTAINS, name=self.name, details=details)
        details["match"] = "none"
        details["reason"] = "no-match"
        return ScorerResult(score=MISS, name=self.name, details=details)


register_scorer_factory(InstructionFollowingScorer.name, InstructionFollowingScorer)
