"""Run graph model with multiple solvers (Gurobi, SCIP, SAT, CBC) N times.

Phases:
- Phase 1: path selection without martes_martes (runs with all solvers).
- Phase 2: path selection for martes_martes using pruned graph from best phase 1 solution.
- Phase 3: rehabilitation using best phase 2 solution.

Results:
- CSV at data/solutions/graph_model_solver_experiment.csv with columns
  ID, SOLVER, PHASE, RUN, TIME, Z, STATUS, SUMMARY, NOTES.
- JSON summaries per solver at data/solutions/summaries/graph_model_experiment/.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import geopandas as gpd
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
RESULTS_PATH = ROOT / "data" / "solutions" / "graph_model_solver_experiment.csv"
SUMMARY_DIR = ROOT / "data" / "solutions" / "summaries" / "graph_model_experiment"
SOLVERS = ["gurobi", "scip", "sat", "cbc"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TOTAL_BUDGET = 500.0
CORRIDOR_SHARE_BY_SPECIES: dict[str, float] = {
    "oryctolagus_cuniculus": 0.30,
    "eliomys_quercinus": 0.24,
    "atelerix_algirus": 0.14,
    "martes_martes": 0.12,
}
ADAPTATION_SHARE_BY_SPECIES: dict[str, float] | None = {
    "oryctolagus_cuniculus": 0.07,
    "eliomys_quercinus": 0.06,
    "atelerix_algirus": 0.04,
    "martes_martes": 0.03,
}

import models.graph_model_gurobi_budgeted as gurobi_model
import models.graph_model_ortools_budgeted as ortools_model
from models.graph_core import (
    CorredorEcologicoMenorca,
    PathCandidate,
    build_cost_dicts,
    build_summary_from_paths,
    collect_used_cells,
    enumerate_candidates,
)
from models.utils import get_adjacent_cells
from models.visualization import save_solution_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run graph model with Gurobi, SCIP, SAT and CBC N times per phase."
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=1,
        help="Number of repetitions per phase and solver.",
    )
    return parser.parse_args()


def load_inputs() -> dict[str, Any]:
    data_path = ROOT / "data" / "processed_dataset.parquet"
    df = gpd.read_parquet(data_path)

    origin_cells_by_species: dict[str, list[str]] = {}
    for species in gurobi_model.SPECIES:
        col = f"has_{species}"
        origin_cells_by_species[species] = df[df[col]]["grid_id"].tolist()

    all_cells = df["grid_id"].tolist()
    adjacency = {
        cell: get_adjacent_cells(cell, set(all_cells), df) for cell in all_cells
    }
    cost_corridor_dict, cost_adaptation_dict, benefit_adaptation_dict = (
        build_cost_dicts(df, all_cells, gurobi_model.SPECIES)
    )

    return {
        "df": df,
        "origin_cells_by_species": origin_cells_by_species,
        "all_cells": all_cells,
        "adjacency": adjacency,
        "cost_corridor_dict": cost_corridor_dict,
        "cost_adaptation_dict": cost_adaptation_dict,
        "benefit_adaptation_dict": benefit_adaptation_dict,
    }


def build_id(solver: str, phase: str, run_idx: int) -> str:
    return f"{phase}_{solver}_run{run_idx}"


def compute_excluded_cells(
    origin_cells_by_species: dict[str, list[str]], species: list[str]
) -> set[str]:
    without_martes = [sp for sp in species if sp != "martes_martes"]
    return {
        cell
        for cell in origin_cells_by_species.get("martes_martes", [])
        if all(cell not in origin_cells_by_species.get(sp, []) for sp in without_martes)
    }


def cells_to_remove_for_phase2(
    selected_phase1: dict[str, list[PathCandidate]],
    origin_cells_by_species: dict[str, list[str]],
) -> set[str]:
    cells_to_remove: set[str] = set(
        cell
        for cell in origin_cells_by_species.get("oryctolagus_cuniculus", [])
        if cell not in origin_cells_by_species.get("martes_martes", [])
    )
    cells_to_remove.update(
        cell
        for cell in origin_cells_by_species.get("eliomys_quercinus", [])
        if cell not in origin_cells_by_species.get("martes_martes", [])
    )
    for species in ["oryctolagus_cuniculus", "eliomys_quercinus"]:
        for candidate in selected_phase1.get(species, []):
            cells_to_remove.update(candidate.cells[1:-1])
    return cells_to_remove


def compute_budget_shares(
    species_list: list[str],
    corridor_shares: dict[str, float],
) -> tuple[float, dict[str, float], dict[str, float], float, float]:
    """Compute per-species budgets from global percentages."""
    corridor_share_total = sum(corridor_shares.values())
    if corridor_share_total > 1.0 + 1e-9:
        raise ValueError(
            f"Corridor shares sum to {corridor_share_total:.3f} > 1.0; "
            "reduce CORRIDOR_SHARE_BY_SPECIES."
        )

    if ADAPTATION_SHARE_BY_SPECIES is None:
        remaining = max(0.0, 1.0 - corridor_share_total)
        adaptation_shares = {sp: remaining / len(species_list) for sp in species_list}
    else:
        adaptation_shares = {
            sp: ADAPTATION_SHARE_BY_SPECIES.get(sp, 0.0) for sp in species_list
        }
        adaptation_share_total = sum(adaptation_shares.values())
        if corridor_share_total + adaptation_share_total > 1.0 + 1e-9:
            raise ValueError(
                f"Corridor+adaptation shares sum to "
                f"{corridor_share_total + adaptation_share_total:.3f} > 1.0; "
                "reduce percentages."
            )

    adaptation_share_total = sum(adaptation_shares.values())
    corridor_budget_total = TOTAL_BUDGET * corridor_share_total
    corridor_budget_by_species = {
        sp: TOTAL_BUDGET * share for sp, share in corridor_shares.items()
    }
    adaptation_budget_by_species = {
        sp: TOTAL_BUDGET * share for sp, share in adaptation_shares.items()
    }

    return (
        corridor_budget_total,
        corridor_budget_by_species,
        adaptation_budget_by_species,
        corridor_share_total,
        adaptation_share_total,
    )


def pick_best_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise RuntimeError("No runs available to select base solution.")
    feasible = [r for r in runs if r.get("objective") is not None]
    target = feasible if feasible else runs
    return min(target, key=lambda r: float(r.get("objective", float("inf"))))


def compute_rehab_objective(
    rehab_selected: dict[str, set[str]],
    cost_adaptation_dict: dict[tuple[str, str], float],
    benefit_adaptation_dict: dict[tuple[str, str], float],
    alpha: float,
) -> float:
    cost = sum(
        cost_adaptation_dict[(sp, cell)]
        for sp, cells in rehab_selected.items()
        for cell in cells
    )
    benefit = sum(
        benefit_adaptation_dict[(sp, cell)]
        for sp, cells in rehab_selected.items()
        for cell in cells
    )
    return alpha * cost - (1 - alpha) * benefit


def add_record(
    records: list[dict[str, Any]],
    *,
    solver: str,
    phase: str,
    run_idx: int,
    objective: float | None,
    runtime: float | None,
    status: Any,
    summary_path: pathlib.Path | None = None,
    notes: str | None = None,
) -> None:
    records.append(
        {
            "ID": build_id(solver, phase, run_idx),
            "SOLVER": solver,
            "PHASE": phase,
            "RUN": run_idx,
            "TIME": runtime,
            "Z": objective,
            "STATUS": status,
            "SUMMARY": str(summary_path) if summary_path else "",
            "NOTES": notes or "",
        }
    )


def mark_canonical(
    records: list[dict[str, Any]], solver: str, phase: str, run_idx: int, note: str
) -> None:
    for rec in records:
        if rec["SOLVER"] == solver and rec["PHASE"] == phase and rec["RUN"] == run_idx:
            rec["NOTES"] = f"{rec['NOTES']}; {note}" if rec["NOTES"] else note
            break


def run_path_selection_gurobi(
    phase_label: str,
    run_idx: int,
    candidates: list[PathCandidate],
    origin_to_paths: dict[tuple[str, str], set[str]],
    species_path_budget: dict[str, float] | None = None,
) -> dict[str, Any]:
    result = gurobi_model.solve_path_selection(
        candidates,
        origin_to_paths,
        phase_label=phase_label,
        species_path_budget=species_path_budget,
        output=False,
    )
    model = result.get("model")
    return {
        "solver": "gurobi",
        "phase": phase_label,
        "run_idx": run_idx,
        "selected": result.get("selected_by_species", {}),
        "objective": float(model.ObjVal)
        if model is not None and model.SolCount > 0
        else None,
        "runtime": float(model.Runtime) if model is not None else None,
        "status": getattr(model, "Status", None),
    }


ORTOOLS_SOLVER_MAP = {
    "scip": "SCIP",
    "cbc": "CBC_MIXED_INTEGER_PROGRAMMING",
    "sat": "SAT",
}


def run_path_selection_ortools(
    solver_id: str,
    solver_name: str,
    phase_label: str,
    run_idx: int,
    candidates: list[PathCandidate],
    origin_to_paths: dict[tuple[str, str], set[str]],
    species_path_budget: dict[str, float] | None = None,
) -> dict[str, Any]:
    result = ortools_model.solve_path_selection_ortools(
        candidates,
        origin_to_paths,
        phase_label=phase_label,
        solver_name=solver_name,
        species_path_budget=species_path_budget,
        enable_output=False,
    )
    solver = result.get("solver")
    runtime = solver.WallTime() / 1000.0 if solver is not None else None
    return {
        "solver": solver_id,
        "phase": phase_label,
        "run_idx": run_idx,
        "selected": result.get("selected_by_species", {}),
        "objective": float(result.get("objective_value"))
        if result.get("objective_value") is not None
        else None,
        "runtime": runtime,
        "status": result.get("status"),
    }


def run_rehab_gurobi(
    run_idx: int,
    used_cells_by_species: dict[str, set[str]],
    cost_corridor_dict: dict[str, float],
    cost_adaptation_dict: dict[tuple[str, str], float],
    benefit_adaptation_dict: dict[tuple[str, str], float],
    origin_cells_by_species: dict[str, list[str]],
    adjacency: dict[str, list[str]],
    all_cells: list[str],
    rehab_budget_by_species: dict[str, float],
) -> dict[str, Any]:
    details = gurobi_model.solve_rehab(
        used_cells_by_species,
        cost_corridor_dict,
        cost_adaptation_dict,
        benefit_adaptation_dict,
        adjacency=adjacency,
        all_cells=all_cells,
        origin_cells_by_species=origin_cells_by_species,
        rehab_budget_by_species=rehab_budget_by_species,
        output=False,
        return_details=True,
    )
    rehab_selected = details.get("rehab_selected", {})
    objective = details.get("objective_value")
    if objective is None:
        objective = compute_rehab_objective(
            rehab_selected,
            cost_adaptation_dict,
            benefit_adaptation_dict,
            gurobi_model.ALPHA,
        )
    return {
        "solver": "gurobi",
        "phase": "fase3_rehabilitacion",
        "run_idx": run_idx,
        "selected": rehab_selected,
        "objective": float(objective) if objective is not None else None,
        "runtime": details.get("runtime_seconds"),
        "status": details.get("status"),
    }


def run_rehab_ortools(
    solver_id: str,
    solver_name: str,
    run_idx: int,
    used_cells_by_species: dict[str, set[str]],
    cost_corridor_dict: dict[str, float],
    cost_adaptation_dict: dict[tuple[str, str], float],
    benefit_adaptation_dict: dict[tuple[str, str], float],
    origin_cells_by_species: dict[str, list[str]],
    adjacency: dict[str, list[str]],
    all_cells: list[str],
    rehab_budget_by_species: dict[str, float],
) -> dict[str, Any]:
    details = ortools_model.solve_rehabilitation_ortools(
        used_cells_by_species,
        cost_corridor_dict,
        cost_adaptation_dict,
        benefit_adaptation_dict,
        origin_cells_by_species,
        adjacency=adjacency,
        all_cells=all_cells,
        solver_name=solver_name,
        rehab_budget_by_species=rehab_budget_by_species,
        enable_output=False,
        return_details=True,
    )
    rehab_selected = details.get("rehab_selected", {})
    objective = details.get("objective_value")
    if objective is None:
        objective = compute_rehab_objective(
            rehab_selected,
            cost_adaptation_dict,
            benefit_adaptation_dict,
            ortools_model.ALPHA,
        )
    return {
        "solver": solver_id,
        "phase": "fase3_rehabilitacion",
        "run_idx": run_idx,
        "selected": rehab_selected,
        "objective": float(objective) if objective is not None else None,
        "runtime": details.get("runtime_seconds"),
        "status": details.get("status"),
    }


def main() -> None:
    args = parse_args()
    inputs = load_inputs()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_time = time.time()

    (
        corridor_budget_total,
        corridor_budget_by_species,
        adaptation_budget_by_species,
        corridor_share_total,
        adaptation_share_total,
    ) = compute_budget_shares(gurobi_model.SPECIES, CORRIDOR_SHARE_BY_SPECIES)

    print(
        f"\nBudgets: total={TOTAL_BUDGET:.2f}, "
        f"corridors share={corridor_share_total:.2f} -> {corridor_budget_total:.2f}, "
        f"adaptation share={adaptation_share_total:.2f}"
    )
    for sp in gurobi_model.SPECIES:
        print(
            f"  {sp}: corridors {corridor_budget_by_species.get(sp, 0.0):.2f} | "
            f"adaptation {adaptation_budget_by_species.get(sp, 0.0):.2f}"
        )

    without_martes = [sp for sp in gurobi_model.SPECIES if sp != "martes_martes"]
    excluded_cells = compute_excluded_cells(
        inputs["origin_cells_by_species"], gurobi_model.SPECIES
    )

    species_path_budget_phase1 = {
        sp: corridor_budget_by_species.get(sp, 0.0) for sp in without_martes
    }
    species_path_budget_phase2 = {
        "martes_martes": corridor_budget_by_species.get("martes_martes", 0.0)
    }

    graph_phase1 = CorredorEcologicoMenorca(
        costes=inputs["cost_corridor_dict"],
        adjacency=inputs["adjacency"],
        gdf=inputs["df"],
        origenes=inputs["origin_cells_by_species"],
    )
    graph_phase1.construir_grafo(excluded_cells=excluded_cells)
    candidates_phase1, _, origin_to_paths_p1 = enumerate_candidates(
        graph_phase1, without_martes, gurobi_model.MAX_DISTANCE_METHOD
    )

    records: list[dict[str, Any]] = []
    phase1_runs: list[dict[str, Any]] = []
    for solver in SOLVERS:
        for run_idx in range(1, args.runs + 1):
            if time.time() - init_time > 300:
                init_time = time.time()
                time.sleep(20)
            try:
                if solver == "gurobi":
                    res = run_path_selection_gurobi(
                        "fase1_sin_martes_martes",
                        run_idx,
                        candidates_phase1,
                        origin_to_paths_p1,
                        species_path_budget=species_path_budget_phase1,
                    )
                else:
                    res = run_path_selection_ortools(
                        solver,
                        ORTOOLS_SOLVER_MAP[solver],
                        "fase1_sin_martes_martes",
                        run_idx,
                        candidates_phase1,
                        origin_to_paths_p1,
                        species_path_budget=species_path_budget_phase1,
                    )
                phase1_runs.append(res)
                add_record(
                    records,
                    solver=res["solver"],
                    phase="phase1",
                    run_idx=run_idx,
                    objective=res["objective"],
                    runtime=res["runtime"],
                    status=res["status"],
                )
            except Exception as exc:  # noqa: BLE001
                add_record(
                    records,
                    solver=solver,
                    phase="phase1",
                    run_idx=run_idx,
                    objective=None,
                    runtime=None,
                    status="error",
                    notes=f"Error: {exc}",
                )

    canonical_phase1 = pick_best_run(phase1_runs)
    canonical_phase1_selected = canonical_phase1.get("selected", {})
    mark_canonical(
        records,
        canonical_phase1.get("solver", ""),
        "phase1",
        canonical_phase1.get("run_idx", 0),
        "Base solution for phase 2",
    )

    cells_to_remove = cells_to_remove_for_phase2(
        canonical_phase1_selected, inputs["origin_cells_by_species"]
    )
    graph_phase2 = CorredorEcologicoMenorca(
        costes=inputs["cost_corridor_dict"],
        adjacency=inputs["adjacency"],
        gdf=inputs["df"],
        origenes=inputs["origin_cells_by_species"],
    )
    graph_phase2.construir_grafo(excluded_cells=cells_to_remove)
    candidates_martes, _, origin_to_paths_martes = enumerate_candidates(
        graph_phase2,
        ["martes_martes"],
        gurobi_model.MAX_DISTANCE_METHOD,
    )

    phase2_runs: list[dict[str, Any]] = []
    for solver in SOLVERS:
        for run_idx in range(1, args.runs + 1):
            if time.time() - init_time > 300:
                init_time = time.time()
                time.sleep(20)
            try:
                if solver == "gurobi":
                    res = run_path_selection_gurobi(
                        "fase2_solo_martes_martes",
                        run_idx,
                        candidates_martes,
                        origin_to_paths_martes,
                        species_path_budget=species_path_budget_phase2,
                    )
                else:
                    res = run_path_selection_ortools(
                        solver,
                        ORTOOLS_SOLVER_MAP[solver],
                        "fase2_solo_martes_martes",
                        run_idx,
                        candidates_martes,
                        origin_to_paths_martes,
                        species_path_budget=species_path_budget_phase2,
                    )
                phase2_runs.append(res)
                add_record(
                    records,
                    solver=res["solver"],
                    phase="phase2",
                    run_idx=run_idx,
                    objective=res["objective"],
                    runtime=res["runtime"],
                    status=res["status"],
                )
            except Exception as exc:  # noqa: BLE001
                add_record(
                    records,
                    solver=solver,
                    phase="phase2",
                    run_idx=run_idx,
                    objective=None,
                    runtime=None,
                    status="error",
                    notes=f"Error: {exc}",
                )

    canonical_phase2 = pick_best_run(phase2_runs)
    canonical_phase2_selected = canonical_phase2.get("selected", {})
    mark_canonical(
        records,
        canonical_phase2.get("solver", ""),
        "phase2",
        canonical_phase2.get("run_idx", 0),
        "Base solution for rehabilitation",
    )

    used_cells_by_species = collect_used_cells(
        canonical_phase1_selected,
        canonical_phase2_selected,
        species_list=gurobi_model.SPECIES,
    )

    summary_phase2 = build_summary_from_paths(
        canonical_phase1_selected,
        canonical_phase2_selected,
        inputs["origin_cells_by_species"],
        species_list=gurobi_model.SPECIES,
        rehab_selected=None,
    )
    phase2_summary_path = SUMMARY_DIR / "phase2_paths_shared.json"
    save_solution_summary(summary_phase2, phase2_summary_path)
    add_record(
        records,
        solver="shared",
        phase="phase2_summary",
        run_idx=0,
        objective=None,
        runtime=None,
        status="summary",
        summary_path=phase2_summary_path,
        notes="Intermediate summary without rehabilitation.",
    )

    for solver in SOLVERS:
        for run_idx in range(1, args.runs + 1):
            if time.time() - init_time > 300:
                init_time = time.time()
                time.sleep(20)
            try:
                if solver == "gurobi":
                    res = run_rehab_gurobi(
                        run_idx,
                        used_cells_by_species,
                        inputs["cost_corridor_dict"],
                        inputs["cost_adaptation_dict"],
                        inputs["benefit_adaptation_dict"],
                        inputs["origin_cells_by_species"],
                        inputs["adjacency"],
                        inputs["all_cells"],
                        rehab_budget_by_species=adaptation_budget_by_species,
                    )
                else:
                    res = run_rehab_ortools(
                        solver,
                        ORTOOLS_SOLVER_MAP[solver],
                        run_idx,
                        used_cells_by_species,
                        inputs["cost_corridor_dict"],
                        inputs["cost_adaptation_dict"],
                        inputs["benefit_adaptation_dict"],
                        inputs["origin_cells_by_species"],
                        inputs["adjacency"],
                        inputs["all_cells"],
                        rehab_budget_by_species=adaptation_budget_by_species,
                    )

                summary_path = (
                    SUMMARY_DIR / f"phase3_rehab_{res['solver']}_run{run_idx}.json"
                )
                summary = build_summary_from_paths(
                    canonical_phase1_selected,
                    canonical_phase2_selected,
                    inputs["origin_cells_by_species"],
                    species_list=gurobi_model.SPECIES,
                    rehab_selected=res["selected"],
                )
                save_solution_summary(summary, summary_path)
                add_record(
                    records,
                    solver=res["solver"],
                    phase="phase3",
                    run_idx=run_idx,
                    objective=res["objective"],
                    runtime=res["runtime"],
                    status=res["status"],
                    summary_path=summary_path,
                )
            except Exception as exc:  # noqa: BLE001
                add_record(
                    records,
                    solver=solver,
                    phase="phase3",
                    run_idx=run_idx,
                    objective=None,
                    runtime=None,
                    status="error",
                    notes=f"Error: {exc}",
                )

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_PATH, index=False, float_format="%.10f")
    print(f"\nCSV saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
