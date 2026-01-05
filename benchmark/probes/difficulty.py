from __future__ import annotations

from typing import Dict


def compute_difficulty(hops: int, alias_depth: int, time_gap: int, distractors: int) -> Dict[str, int]:
    return {
        "hops": hops,
        "alias_depth": alias_depth,
        "time_gap": time_gap,
        "distractors": distractors,
    }


