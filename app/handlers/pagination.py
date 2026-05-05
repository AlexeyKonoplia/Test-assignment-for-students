"""Helpers for list responses returned by handlers."""

from __future__ import annotations

from collections.abc import Iterable


DEFAULT_LIMIT = 1000


def limited(items: Iterable[dict], total_available: int, limit: int = DEFAULT_LIMIT) -> dict:
    """Return pagination metadata and a materialized first page."""
    page = list(items)
    return {
        "limit": int(limit),
        "total_available": int(total_available),
        "truncated": int(total_available) > int(limit),
        "items": page,
    }
