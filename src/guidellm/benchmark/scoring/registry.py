"""Name-based scorer registry with instance and factory registration."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .protocol import Scorer

__all__ = [
    "register_scorer",
    "register_scorer_factory",
    "get_scorer",
    "list_scorers",
]

_lock = threading.Lock()
_instances: dict[str, Scorer] = {}
_factories: dict[str, Callable[..., Scorer]] = {}


def register_scorer(name: str, scorer: Scorer) -> None:
    """Register a ready-made scorer instance under ``name``."""
    with _lock:
        _instances[name] = scorer


def register_scorer_factory(name: str, factory: Callable[..., Scorer]) -> None:
    """Register a factory so ``get_scorer(name, **kwargs)`` builds a fresh
    scorer per benchmark run (e.g. to inject a per-run sentinel)."""
    with _lock:
        _factories[name] = factory


def get_scorer(name: str, **kwargs: Any) -> Scorer:
    """Return the scorer registered under ``name``.

    :raises KeyError: if no scorer or factory is registered under ``name``.
    :raises ValueError: if kwargs are given but only an instance is registered.
    """
    with _lock:
        factory = _factories.get(name)
        instance = _instances.get(name)
    if factory is not None:
        return factory(**kwargs)
    if instance is not None:
        if kwargs:
            raise ValueError(
                f"scorer {name} is a shared instance and takes no config kwargs"
            )
        return instance
    raise KeyError(f"unknown scorer: {name!r} (registered: {list_scorers()})")


def list_scorers() -> list[str]:
    """Return all registered scorer names (instances and factories)."""
    with _lock:
        return sorted(set(_instances) | set(_factories))
