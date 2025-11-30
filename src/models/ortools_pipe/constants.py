from __future__ import annotations

"""
Shared constants for the pipeline models.
Keeping defaults centralized makes solver configuration easier to follow.
"""

SPECIES = [
    "oryctolagus_cuniculus",
    "atelerix_algirus",
    "eliomys_quercinus",
    "martes_martes",
]

BUDGET = 500.0
ALPHA = 0.5
TIME_LIMIT_MS = 1_000 * 60 * 10
MAX_DISTANCE = 20
NUM_THREADS = 10
