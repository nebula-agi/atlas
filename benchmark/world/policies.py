from __future__ import annotations

from typing import List

from .schema import MemoryPolicy


def default_policies() -> List[MemoryPolicy]:
    """
    Baseline memory decay policies. These can be extended per domain.
    """
    return [
        MemoryPolicy(fact_predicate="deployed_in", decay_half_life=60, retrievable_after_expiry=True),
        MemoryPolicy(fact_predicate="impact", decay_half_life=45, retrievable_after_expiry=False),
        MemoryPolicy(fact_predicate="led", decay_half_life=120, retrievable_after_expiry=True),
    ]


