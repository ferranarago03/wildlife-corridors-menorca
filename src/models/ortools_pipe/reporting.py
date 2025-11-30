from __future__ import annotations

import ortools.linear_solver.pywraplp as pywraplp
import ortools.sat.python.cp_model as cp_model


def print_results(
    solver: pywraplp.Solver,
    status: int,
    elapsed_time: float,
    phase: str,
    cost: float = 0.0,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"SOLUTION RESULTS - {phase}")
    print(f"{'=' * 60}")

    if status == pywraplp.Solver.OPTIMAL:
        print("✓ OPTIMAL solution found!")
    elif status == pywraplp.Solver.FEASIBLE:
        print("✓ FEASIBLE solution found (time limit reached, not proven optimal)")
    elif status == pywraplp.Solver.NOT_SOLVED:
        print(
            "✗ Solver did not run to completion (e.g., time limit before first solution)"
        )
    elif status == pywraplp.Solver.INFEASIBLE:
        print("✗ Problem is INFEASIBLE")
    else:
        print(f"✗ No solution found (Status: {status})")

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        print(f"\nObjective value: {solver.Objective().Value():.2f}")
        if cost > 0:
            print(f"Total cost: {cost:.2f}")
        print(f"Number of variables: {solver.NumVariables()}")
        print(f"Number of constraints: {solver.NumConstraints()}")
        print(f"Solver runtime: {solver.WallTime() / 1000.0:.2f} seconds")
        print(f"Actual elapsed time: {elapsed_time:.2f} seconds")


def print_results_cpsat(
    solver: cp_model.CpSolver,
    status: int,
    elapsed_time: float,
    phase: str,
    *,
    actual_cost: float | None = None,
    objective_scale: float = 1.0,
    num_variables: int | None = None,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"SOLUTION RESULTS - {phase}")
    print(f"{'=' * 60}")

    status_name = solver.StatusName(status)
    print(f"Solver status: {status_name}")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"✗ No feasible solution found (status: {status_name})")
        print(f"Actual elapsed time: {elapsed_time:.2f} seconds")
        return

    objective_value = solver.ObjectiveValue()
    if objective_scale and abs(objective_scale - 1.0) > 1e-9:
        unscaled_objective = objective_value / objective_scale
        print(
            f"Objective value: {unscaled_objective:.4f} (scaled: {objective_value:.2f})"
        )
    else:
        print(f"Objective value: {objective_value:.4f}")

    if actual_cost is not None:
        print(f"Total cost: {actual_cost:.2f}")

    if num_variables is not None:
        print(f"Decision variables (approx.): {num_variables}")

    print(f"Solver runtime: {solver.WallTime():.2f} seconds")
    print(f"Actual elapsed time: {elapsed_time:.2f} seconds")
