"""Composable scorer adapters.

ThinkingBlockStripper removes reasoning blocks from a response before
delegating to an inner scorer, so reasoning models are graded on their
answer, not their chain-of-thought. Adapters compose freely.
"""

from __future__ import annotations

import re
from typing import Any

from .protocol import Scorer, ScorerResult

__all__ = ["strip_thinking_blocks", "ThinkingBlockStripper"]

DEFAULT_TAGS = ("think", "reasoning", "thought", "scratchpad")


def _strip_blocks(text: str, tags: tuple[str, ...]) -> tuple[str, bool]:
    """Remove reasoning blocks. Returns (cleaned, any_block_removed)."""
    stripped_any = False
    stripped = text
    for tag in tags:
        # Paired spans first (non-greedy, DOTALL, case-insensitive)
        new, n = re.subn(
            r"<" + tag + r"\b[^>]*>.*?</" + tag + r"\s*>",
            "",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        stripped, stripped_any = new, stripped_any or n > 0
        # Unclosed opening tag: strip to end of text
        new, n = re.subn(
            r"<" + tag + r"\b[^>]*>.*$",
            "",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        stripped, stripped_any = new, stripped_any or n > 0
    # Fenced thinking blocks: ```thinking ... ```
    new, n = re.subn(
        r"```\s*thinking\b.*?```",
        "",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    stripped, stripped_any = new, stripped_any or n > 0
    # Collapse leftover blank lines, then trim ends
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped, stripped_any


def strip_thinking_blocks(text: str, tags: tuple[str, ...] = DEFAULT_TAGS) -> str:
    """Remove reasoning blocks from text.

    Handles ``<tag>...</tag>`` spans (case-insensitive, non-greedy), markdown
    fenced `````thinking`` blocks, and unclosed opening tags (stripped to end
    of text). Collapses leftover blank lines. No-op when no blocks present.
    """
    if not text:
        return text
    return _strip_blocks(text, tags)[0]


class ThinkingBlockStripper:
    """Scorer adapter that strips thinking blocks before delegating."""

    def __init__(
        self,
        inner: Scorer,
        tags: tuple[str, ...] = DEFAULT_TAGS,
        name_suffix: str = "_nothink",
    ) -> None:
        self.inner = inner
        self.tags = tags
        self.name = inner.name + name_suffix

    def score(
        self,
        output: str,
        expected: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ScorerResult:
        cleaned, stripped = _strip_blocks(output or "", self.tags)
        result = self.inner.score(cleaned, expected=expected, context=context)
        details = dict(result.details)
        details["stripped"] = stripped
        details["chars_removed"] = len(output or "") - len(cleaned)
        return ScorerResult(score=result.score, name=self.name, details=details)
