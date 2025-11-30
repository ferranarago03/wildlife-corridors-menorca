"""
Small shared helpers for Gurobi models.
Only contains the pieces reused across the two pipeline scripts to keep them light.
"""

from __future__ import annotations

from typing import Optional

import gurobipy as gp
from gurobipy import GRB

from ..utils import get_adjacent_cells, manhattan_distance, parse_grid_id


def create_model(
    name: str,
    *,
    output: bool = True,
    time_limit_seconds: Optional[float],
    gap: float | None = None,
    heuristics: float,
    focus: int,
) -> gp.Model:
    model = gp.Model(name)
    if output:
        model.setParam("OutputFlag", 1)
    else:
        model.setParam("OutputFlag", 0)
    if time_limit_seconds is not None:
        model.setParam("TimeLimit", time_limit_seconds)
    if gap is not None:
        model.setParam("MIPGap", gap)
    model.setParam("MIPFocus", focus)
    model.setParam("Heuristics", heuristics)
    return model


def print_results(model: gp.Model, elapsed_time: float, phase: str, cost: gp.LinExpr):
    """Standardized log of solver status for consistency across scripts."""
    print(f"\n{'=' * 60}")
    print(f"SOLUTION RESULTS - {phase}")
    print(f"{'=' * 60}")
    print(f"Solution status: {model.Status}")

    if model.Status == GRB.OPTIMAL:
        print("✓ OPTIMAL solution found!")
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        print("✓ FEASIBLE solution found (time limit reached, not proven optimal)")
    elif model.Status in [GRB.SUBOPTIMAL]:
        print("✓ FEASIBLE solution found (not proven optimal)")
    else:
        print("✗ No solution found")

    if model.SolCount > 0:
        print(f"\nObjective value: {model.ObjVal:.2f}")
        print(f"Total cost: {cost.getValue():.2f}")
        print(f"Number of variables: {model.NumVars}")
        print(f"Number of constraints: {model.NumConstrs}")
        print(f"Solver runtime: {model.Runtime:.2f} seconds")
        print(f"Actual elapsed time: {elapsed_time:.2f} seconds")


__all__ = [
    "create_model",
    "print_results",
    "get_adjacent_cells",
    "manhattan_distance",
    "parse_grid_id",
]
