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


def _remove_blocks(text: str, tags: tuple[str, ...]) -> tuple[str, bool]:
    """Remove reasoning-block markup only. Returns (cleaned, any_removed).

    Handles ``<tag>...</tag>`` spans (case-insensitive), true nesting
    (innermost pairs removed first via a tempered pattern, repeated to a
    fixpoint), orphan closing tags left by malformed nesting, unclosed
    opening tags (stripped to end of text), and fenced `````thinking```
    blocks. No whitespace normalization is applied here.
    """
    stripped_any = False
    stripped = text
    for tag in tags:
        # Innermost paired spans first: the tempered content cannot contain
        # another open/close of the same tag, so nested spans collapse
        # inside-out. Repeated to a fixpoint.
        inner = (
            r"<" + tag + r"\b[^>]*>"
            r"(?:(?!</?" + tag + r"\b).)*?"
            r"</" + tag + r"\s*>"
        )
        while True:
            new, n = re.subn(inner, "", stripped, flags=re.IGNORECASE | re.DOTALL)
            stripped_any = stripped_any or n > 0
            if n == 0:
                break
            stripped = new
        # Orphan closing tags (malformed/nesting residue): remove the tag.
        new, n = re.subn(
            r"</" + tag + r"\s*>", "", stripped, flags=re.IGNORECASE
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
    while True:
        new, n = re.subn(
            r"```\s*thinking\b.*?```",
            "",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        stripped_any = stripped_any or n > 0
        if n == 0:
            break
        stripped = new
    return stripped, stripped_any


def _strip_blocks(text: str, tags: tuple[str, ...]) -> tuple[str, bool]:
    """Remove reasoning blocks and normalize whitespace.

    Returns (cleaned, any_block_removed). Block removal is followed by
    collapsing 3+ newlines to two and trimming the ends (needed so graded
    answers match sentinels exactly). When no blocks are present the text
    is returned whitespace-normalized but otherwise unchanged.
    """
    stripped, stripped_any = _remove_blocks(text, tags)
    # Collapse leftover blank lines, then trim ends
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped, stripped_any


def strip_thinking_blocks(text: str, tags: tuple[str, ...] = DEFAULT_TAGS) -> str:
    """Remove reasoning blocks from text.

    Handles ``<tag>...</tag>`` spans (case-insensitive, non-greedy), markdown
    fenced `````thinking`` blocks, and unclosed opening tags (stripped to end
    of text). Also collapses 3+ newlines and trims the ends.
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
        raw = output or ""
        block_cleaned, stripped = _remove_blocks(raw, self.tags)
        # chars_removed measures block markup only, not whitespace cleanup.
        chars_removed = len(raw) - len(block_cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", block_cleaned).strip()
        result = self.inner.score(cleaned, expected=expected, context=context)
        details = dict(result.details)
        details["stripped"] = stripped
        details["chars_removed"] = chars_removed
        return ScorerResult(score=result.score, name=self.name, details=details)
