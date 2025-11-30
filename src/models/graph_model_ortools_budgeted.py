from __future__ import annotations

import pathlib
import sys
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

import geopandas as gpd
from ortools.linear_solver import pywraplp

# Ensure src is on the path so `models.*` imports work both as module and script
src_path = pathlib.Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from models.graph_core import (
    MenorcaEcologicalCorridor,
    PathCandidate,
    _compute_corridor_cost,
    build_cost_dicts,
    build_summary_from_paths,
    collect_used_cells,
    enumerate_candidates,
    report_selection,
)
from models.ortools_pipe.constants import (
    ALPHA,
    MAX_DISTANCE,
    NUM_THREADS,
    SPECIES,
    TIME_LIMIT_MS,
)
from models.ortools_pipe.reporting import print_results
from models.utils import get_adjacent_cells
from models.visualization import create_solution_map, save_solution_summary

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_DISTANCE_METHOD = min(15, MAX_DISTANCE)
ORTOOLS_SOLVER_NAME = "SCIP"
PENALTY_UNCOVERED_ORIGIN: float | None = 1e3
MIN_COVERAGE_FRACTION: float | None = None

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


def _validate_and_prepare_budget_shares(
    species_list: Sequence[str],
) -> tuple[float, dict[str, float], dict[str, float], float, float]:
    corridor_shares = {
        sp: CORRIDOR_SHARE_BY_SPECIES.get(sp, 0.0) for sp in species_list
    }
    corridor_share_total = sum(corridor_shares.values())
    if corridor_share_total > 1.0 + 1e-9:
        raise ValueError(
            f"Corridor shares sum to {corridor_share_total:.3f} > 1.0; "
            "reduce CORRIDOR_SHARE_BY_SPECIES."
        )

    if ADAPTATION_SHARE_BY_SPECIES is None:
        remaining = max(0.0, 1.0 - corridor_share_total)
        adaptation_shares = {sp: remaining / len(species_list) for sp in species_list}
        adaptation_share_total = sum(adaptation_shares.values())
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


# ---------------------------------------------------------------------------
# OR-Tools helpers
# ---------------------------------------------------------------------------
def _create_mip_solver(
    solver_name: str,
    *,
    time_limit_ms: int | None,
    num_threads: int,
    enable_output: bool = True,
) -> pywraplp.Solver:
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(
            f"The {solver_name} backend is not available in this OR-Tools installation."
        )
    if time_limit_ms is not None:
        solver.SetTimeLimit(int(time_limit_ms))
    solver.SetNumThreads(max(1, int(num_threads)))
    if enable_output:
        solver.EnableOutput()
    else:
        solver.SuppressOutput()
    print(
        f"Using OR-Tools solver: {solver_name}, "
        f"time_limit_ms={time_limit_ms}, threads={num_threads}"
    )
    return solver


def solve_path_selection_ortools(
    candidates: Sequence[PathCandidate],
    origin_to_paths: Mapping[tuple[str, str], set[str]],
    *,
    phase_label: str,
    solver_name: str = ORTOOLS_SOLVER_NAME,
    time_limit_ms: int | None = TIME_LIMIT_MS,
    num_threads: int = NUM_THREADS,
    enable_output: bool = True,
    species_path_budget: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """ILP model using OR-Tools with per-species budget."""
    if not candidates:
        print(f"\n{phase_label}: no candidates to optimize.")
        return {"selected_by_species": {}}

    print(
        f"{phase_label}: config -> time_limit_ms={time_limit_ms}, num_threads={num_threads}, "
        f"penalty_uncovered_origin={PENALTY_UNCOVERED_ORIGIN}, "
        f"min_coverage_fraction={MIN_COVERAGE_FRACTION}"
    )

    solver = _create_mip_solver(
        solver_name,
        time_limit_ms=time_limit_ms,
        num_threads=num_threads,
        enable_output=enable_output,
    )

    z_vars: dict[str, pywraplp.Variable] = {}
    objective_terms = []

    for candidate in candidates:
        var = solver.BoolVar(f"z_{candidate.path_id}")
        z_vars[candidate.path_id] = var
        objective_terms.append(candidate.cost * var)

    use_optional_coverage = (
        PENALTY_UNCOVERED_ORIGIN is not None or MIN_COVERAGE_FRACTION is not None
    )
    coverage_vars: dict[tuple[str, str], pywraplp.Variable] = {}

    if use_optional_coverage:
        for key in origin_to_paths.keys():
            coverage_vars[key] = solver.BoolVar(f"covered_{key[0]}_{key[1]}")

    for (species, origin), path_ids in origin_to_paths.items():
        coverage_var = coverage_vars.get((species, origin))
        if not path_ids:
            if coverage_var is not None:
                solver.Add(coverage_var == 0)
            continue
        path_sum = solver.Sum(z_vars[pid] for pid in path_ids)
        if coverage_var is not None:
            solver.Add(path_sum >= coverage_var, f"origin_{species}_{origin}")
        else:
            solver.Add(path_sum >= 1, f"origin_{species}_{origin}")

    if MIN_COVERAGE_FRACTION is not None and coverage_vars:
        required_coverage = MIN_COVERAGE_FRACTION * len(coverage_vars)
        solver.Add(solver.Sum(coverage_vars.values()) >= required_coverage)

    if PENALTY_UNCOVERED_ORIGIN is not None and coverage_vars:
        objective_terms.extend(
            PENALTY_UNCOVERED_ORIGIN * (1 - var) for var in coverage_vars.values()
        )

    if species_path_budget:
        for species, cap in species_path_budget.items():
            species_candidates = [c for c in candidates if c.species == species]
            if not species_candidates or cap is None:
                continue
            solver.Add(
                solver.Sum(c.cost * z_vars[c.path_id] for c in species_candidates)
                <= cap
            )

    solver.Minimize(solver.Sum(objective_terms))

    start_time = time.time()
    status = solver.Solve()
    elapsed_time = time.time() - start_time
    total_cost = (
        solver.Objective().Value()
        if status
        in (
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
        )
        else 0.0
    )

    print_results(solver, status, elapsed_time, phase_label, total_cost)

    selected_candidates: list[PathCandidate] = []
    selected_by_species: dict[str, list[PathCandidate]] = defaultdict(list)

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for candidate in candidates:
            if z_vars[candidate.path_id].solution_value() > 0.5:
                selected_candidates.append(candidate)
                selected_by_species[candidate.species].append(candidate)

    print(
        f"{phase_label}: selected {len(selected_candidates)} paths "
        f"from {len(candidates)} candidates."
    )

    return {
        "solver": solver,
        "z": z_vars,
        "selected_candidates": selected_candidates,
        "selected_by_species": dict(selected_by_species),
        "objective_value": total_cost,
        "status": status,
    }


def solve_rehabilitation_ortools(
    used_cells_by_species: Mapping[str, set[str]],
    cost_corridor_dict: Mapping[str, float],
    cost_adaptation_dict: Mapping[tuple[str, str], float],
    benefit_adaptation_dict: Mapping[tuple[str, str], float],
    origin_cells_by_species: Mapping[str, list[str]],
    *,
    adjacency: Mapping[str, Sequence[str]],
    all_cells: Sequence[str],
    alpha: float = ALPHA,
    solver_name: str = ORTOOLS_SOLVER_NAME,
    time_limit_ms: int | None = TIME_LIMIT_MS,
    num_threads: int = NUM_THREADS,
    phase_label: str = "rehabilitation_post_paths",
    rehab_budget_by_species: Mapping[str, float] | None = None,
    enable_output: bool = True,
    return_details: bool = False,
) -> dict[str, Any]:
    """OR-Tools model for selecting cells to rehabilitate after corridors."""
    solver = _create_mip_solver(
        solver_name,
        time_limit_ms=time_limit_ms,
        num_threads=num_threads,
        enable_output=enable_output,
    )

    rehab_vars: dict[tuple[str, str], pywraplp.Variable] = {}
    candidate_cells_by_species: dict[str, set[str]] = {}

    def _cell_allowed_for_species(species: str, cell: str) -> bool:
        """Avoid creating variables in cells incompatible with constraints."""
        if species == "martes_martes":
            return cell not in used_cells_by_species.get(
                "oryctolagus_cuniculus", set()
            ) and (cell not in used_cells_by_species.get("eliomys_quercinus", set()))
        if species in {"oryctolagus_cuniculus", "eliomys_quercinus"}:
            return cell not in used_cells_by_species.get("martes_martes", set())
        return True

    for species in SPECIES:
        origins = origin_cells_by_species.get(species, [])
        corridor_cells = used_cells_by_species.get(species, set())
        candidates: set[str] = set()
        for cell in corridor_cells:
            for neighbor in adjacency.get(cell, []):
                if neighbor not in origins and _cell_allowed_for_species(
                    species, neighbor
                ):
                    candidates.add(neighbor)
        for cell in origins:
            for neighbor in adjacency.get(cell, []):
                if _cell_allowed_for_species(species, neighbor):
                    candidates.add(neighbor)
        candidate_cells_by_species[species] = candidates
        for candidate in candidates:
            rehab_vars[(species, candidate)] = solver.BoolVar(
                f"rehab_{species}_{candidate}"
            )

    if not rehab_vars:
        print("\nNo candidates for rehabilitation.")
        empty = {sp: set() for sp in SPECIES}
        details = {
            "rehab_selected": empty,
            "solver": solver,
            "objective_value": None,
            "runtime_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "status": pywraplp.Solver.INFEASIBLE,
        }
        return details if return_details else empty

    cost_expr = solver.Sum(
        cost_adaptation_dict[(sp, cell)] * var for (sp, cell), var in rehab_vars.items()
    )
    benefit_expr = solver.Sum(
        benefit_adaptation_dict[(sp, cell)] * var
        for (sp, cell), var in rehab_vars.items()
    )

    solver.Minimize(alpha * cost_expr - (1 - alpha) * benefit_expr)

    for cell in all_cells:
        terms = []
        if (var := rehab_vars.get(("martes_martes", cell))) is not None:
            terms.append(2 * var)
        if (var := rehab_vars.get(("oryctolagus_cuniculus", cell))) is not None:
            terms.append(var)
        if (var := rehab_vars.get(("eliomys_quercinus", cell))) is not None:
            terms.append(var)
        if terms:
            solver.Add(solver.Sum(terms) <= 2)

    used_cells = set(cell for cells in used_cells_by_species.values() for cell in cells)
    used_cost = sum(cost_corridor_dict.get(cell, 0.0) for cell in used_cells)
    print(f"\nCost used in corridors: {used_cost:.2f}")

    if rehab_budget_by_species:
        for species, cap in rehab_budget_by_species.items():
            if cap is None or cap <= 0:
                continue
            species_terms = [
                cost_adaptation_dict[(species, cell)] * var
                for (sp, cell), var in rehab_vars.items()
                if sp == species
            ]
            if not species_terms:
                continue
            solver.Add(solver.Sum(species_terms) <= cap)

    start_time = time.time()
    status = solver.Solve()
    elapsed_time = time.time() - start_time
    total_cost = (
        solver.Objective().Value()
        if status
        in (
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
        )
        else 0.0
    )

    print_results(solver, status, elapsed_time, phase_label, total_cost)

    rehab_selected: dict[str, set[str]] = {sp: set() for sp in SPECIES}
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for (species, cell), var in rehab_vars.items():
            if var.solution_value() > 0.5:
                rehab_selected.setdefault(species, set()).add(cell)

    details = {
        "rehab_selected": rehab_selected,
        "solver": solver,
        "objective_value": total_cost
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
        else None,
        "runtime_seconds": solver.WallTime() / 1000.0,
        "elapsed_seconds": elapsed_time,
        "status": status,
    }
    return details if return_details else rehab_selected


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline() -> None:
    """Execute the complete pipeline using OR-Tools with per-species budgets."""
    (
        corridor_budget_total,
        corridor_budget_by_species,
        adaptation_budget_by_species,
        corridor_share_total,
        adaptation_share_total,
    ) = _validate_and_prepare_budget_shares(SPECIES)

    print(
        "\nBudgets configured:"
        f"\n  Total: {TOTAL_BUDGET:.2f}"
        f"\n  Corridors (sum shares): {corridor_share_total:.2f} -> "
        f"{corridor_budget_total:.2f}"
    )
    for sp in SPECIES:
        print(
            f"    {sp}: corridors {corridor_budget_by_species.get(sp, 0.0):.2f} "
            f"| adaptation {adaptation_budget_by_species.get(sp, 0.0):.2f}"
        )
    print(f"  Adaptation (sum shares): {adaptation_share_total:.2f}\n")

    root_path = pathlib.Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "processed_dataset.parquet"
    df = gpd.read_parquet(data_path)

    origin_cells_by_species: dict[str, list[str]] = {}
    for species in SPECIES:
        column_name = f"has_{species}"
        species_cells = df[df[column_name]]["grid_id"].tolist()
        if not species_cells:
            print(f"Warning: No cells found for species {species}")
        origin_cells_by_species[species] = species_cells

    all_cells = df["grid_id"].tolist()
    adjacency = {
        cell: get_adjacent_cells(cell, set(all_cells), df) for cell in all_cells
    }
    (
        cost_corridor_dict,
        cost_adaptation_dict,
        benefit_adaptation_dict,
    ) = build_cost_dicts(df, all_cells, SPECIES)

    without_martes = [species for species in SPECIES if species != "martes_martes"]
    excluded_cells = set(
        cell
        for cell in origin_cells_by_species.get("martes_martes", [])
        if all(cell not in origin_cells_by_species.get(sp, []) for sp in without_martes)
    )

    print("\n====================")
    print("PHASE 1: Corridors without martes_martes")
    print("====================")
    graph_phase1 = MenorcaEcologicalCorridor(
        costs=cost_corridor_dict,
        adjacency=adjacency,
        gdf=df,
        origins=origin_cells_by_species,
    )
    graph_phase1.build_graph(excluded_cells=excluded_cells)

    candidates_phase1, _, origin_to_paths_p1 = enumerate_candidates(
        graph_phase1,
        without_martes,
        MAX_DISTANCE_METHOD,
    )

    species_path_budget_phase1 = {
        sp: corridor_budget_by_species.get(sp, 0.0) for sp in without_martes
    }
    solution_phase1 = solve_path_selection_ortools(
        candidates_phase1,
        origin_to_paths_p1,
        phase_label="sin_martes_martes",
        species_path_budget=species_path_budget_phase1,
    )
    selected_by_species_phase1 = solution_phase1.get("selected_by_species", {})
    cost_phase1 = _compute_corridor_cost(selected_by_species_phase1, cost_corridor_dict)
    print(f"Budget used in phase 1: {cost_phase1:.2f}")

    cells_to_remove = set(
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
        for candidate in selected_by_species_phase1.get(species, []):
            cells_to_remove.update(candidate.cells[1:-1])

    print("\n====================")
    print("PHASE 2: Corridors for martes_martes")
    print("====================")
    graph_phase2 = MenorcaEcologicalCorridor(
        costs=cost_corridor_dict,
        adjacency=adjacency,
        gdf=df,
        origins=origin_cells_by_species,
    )
    graph_phase2.build_graph(excluded_cells=cells_to_remove)

    candidates_martes, _, origin_to_paths_martes = enumerate_candidates(
        graph_phase2,
        ["martes_martes"],
        MAX_DISTANCE_METHOD,
    )

    species_path_budget_phase2 = {
        "martes_martes": corridor_budget_by_species.get("martes_martes", 0.0)
    }
    solution_martes = solve_path_selection_ortools(
        candidates_martes,
        origin_to_paths_martes,
        phase_label="solo_martes_martes",
        species_path_budget=species_path_budget_phase2,
    )
    selected_by_species_martes = solution_martes.get("selected_by_species", {})

    report_selection(
        "Selection phase without martes_martes",
        selected_by_species_phase1,
        species_list=SPECIES,
    )
    report_selection(
        "Selection phase martes_martes",
        selected_by_species_martes,
        species_list=SPECIES,
    )

    print("\n====================")
    print("PHASE 3: Post-corridor rehabilitation")
    print("====================")
    used_cells_by_species = collect_used_cells(
        selected_by_species_phase1, selected_by_species_martes, species_list=SPECIES
    )
    rehab_selected_by_species = solve_rehabilitation_ortools(
        used_cells_by_species,
        cost_corridor_dict,
        cost_adaptation_dict,
        benefit_adaptation_dict,
        origin_cells_by_species,
        adjacency=adjacency,
        all_cells=all_cells,
        rehab_budget_by_species=adaptation_budget_by_species,
    )

    print("\n====================")
    print("PHASE 4: Visualization and export")
    print("====================")
    summary = build_summary_from_paths(
        selected_by_species_phase1,
        selected_by_species_martes,
        origin_cells_by_species,
        species_list=SPECIES,
        rehab_selected=rehab_selected_by_species,
    )
    summary_path = (
        root_path
        / "data"
        / "experiments"
        / "summaries"
        / "graph_model_ortools_budgeted_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_solution_summary(summary, summary_path)

    maps_dir = root_path / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    map_all = create_solution_map(
        df,
        summary,
        species=None,
        title="Corridors (OR-Tools per-species budgets) - All species",
    )
    map_all_path = maps_dir / "graph_model_ortools_budgeted_map_all.html"
    map_all.save(str(map_all_path))
    print(f"General map saved to {map_all_path}")

    for sp in SPECIES:
        map_sp = create_solution_map(
            df,
            summary,
            species=sp,
            title=f"Corridors (OR-Tools per-species budgets) - {sp}",
        )
        map_sp_path = maps_dir / f"graph_model_ortools_budgeted_map_{sp}.html"
        map_sp.save(str(map_sp_path))
        print(f"Map for {sp} saved to {map_sp_path}")


if __name__ == "__main__":
    run_pipeline()
