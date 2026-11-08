"""Retrieval evaluation metrics."""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the first ``k`` retrieved items that are relevant.

    The denominator is the number of items actually returned (at most ``k``), so a
    short result list is not penalized for missing slots. Returns 0.0 for empty
    results or ``k <= 0``.
    """
    if k <= 0:
        return 0.0
    top_items = retrieved[:k]
    if not top_items:
        return 0.0
    hits = sum(1 for item in top_items if item in relevant)
    return hits / len(top_items)
