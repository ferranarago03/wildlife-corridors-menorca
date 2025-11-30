"""
Shared default constants for the Gurobi-based models.
Keeping them in one place helps newcomers see the key knobs quickly.
"""

SPECIES = [
    "oryctolagus_cuniculus",
    "atelerix_algirus",
    "eliomys_quercinus",
    "martes_martes",
]

# Costs and weighting
BUDGET = 500.0
ALPHA = 0.5

# Solver tuning
GAP = 0.05
HEURISTICS = 0.3
FOCUS = 1

# Time limits (minutes expressed in seconds)
TIME_LIMIT_PIPE = 60 * 5  # three-phase pipeline
TIME_LIMIT_METHOD = 60 * 5  # distance-based methodology

# Distance limits
MAX_DISTANCE_PIPE = 20  # skip long arcs in the pipeline model
MAX_DISTANCE_METHOD = 15  # skip long arcs in the methodology model
MAX_MANHATTAN_METHOD = 5  # pruning radius around individual corridors

# Penalties and coverage
PENALTY_UNCOVERED_ORIGIN: float | None = 1e3
MIN_COVERAGE_FRACTION: float | None = None
