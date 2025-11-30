from __future__ import annotations

import pathlib
import sys
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

import geopandas as gpd
import gurobipy as gp

# Allow running as script by ensuring src is on sys.path
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
from models.gurobi import (
    ALPHA,
    FOCUS,
    HEURISTICS,
    MAX_DISTANCE_METHOD,
    MIN_COVERAGE_FRACTION,
    PENALTY_UNCOVERED_ORIGIN,
    SPECIES,
    TIME_LIMIT_METHOD,
    create_model,
    print_results,
)
from models.utils import get_adjacent_cells
from models.visualization import create_solution_map, save_solution_summary

# ---------------------------------------------------------------------------
# Budget configuration
# ---------------------------------------------------------------------------
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

# Solver tuning overrides for this variant (defaults come from models.gurobi)
PENALTY_UNCOVERED_ORIGIN_OVERRIDE = PENALTY_UNCOVERED_ORIGIN
MIN_COVERAGE_FRACTION_OVERRIDE = MIN_COVERAGE_FRACTION


def _validate_and_prepare_budget_shares(
    species_list: Sequence[str],
) -> tuple[float, dict[str, float], dict[str, float], float, float]:
    """Validate percentages and return budgets per species."""
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


def solve_path_selection(
    candidates: Sequence[PathCandidate],
    origin_to_paths: Mapping[tuple[str, str], set[str]],
    *,
    phase_label: str,
    output: bool = True,
    time_limit: float | None = TIME_LIMIT_METHOD,
    heuristics: float = HEURISTICS,
    focus: int = FOCUS,
    penalty_uncovered_origin: float | None = PENALTY_UNCOVERED_ORIGIN_OVERRIDE,
    min_coverage_fraction: float | None = MIN_COVERAGE_FRACTION_OVERRIDE,
    species_path_budget: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """ILP with per-species budgets (path cost)."""
    if not candidates:
        print(f"\n{phase_label}: no candidates to optimize.")
        return {}

    print(
        f"{phase_label}: config -> time_limit={time_limit}, heuristics={heuristics}, "
        f"focus={focus}, penalty_uncovered_origin={penalty_uncovered_origin}, "
        f"min_coverage_fraction={min_coverage_fraction}"
    )
    model = create_model(
        f"path_selection_{phase_label}",
        output=output,
        time_limit_seconds=time_limit,
        heuristics=heuristics,
        focus=focus,
    )

    z: dict[str, gp.Var] = {}
    cost_expr = gp.LinExpr()
    for candidate in candidates:
        var = model.addVar(vtype=gp.GRB.BINARY, name=f"z_{candidate.path_id}")
        z[candidate.path_id] = var
        cost_expr += candidate.cost * var

    origins = list(origin_to_paths.keys())
    use_optional_coverage = (
        penalty_uncovered_origin is not None or min_coverage_fraction is not None
    )

    if use_optional_coverage:
        c_vars: dict[tuple[str, str], gp.Var] = {}
        for (species, origin), path_ids in origin_to_paths.items():
            c_var = model.addVar(vtype=gp.GRB.BINARY, name=f"c_{species}_{origin}")
            c_vars[(species, origin)] = c_var
            if not path_ids:
                model.addConstr(c_var == 0, name=f"no_paths_{species}_{origin}")
                continue
            model.addConstr(
                gp.quicksum(z[pid] for pid in path_ids) >= c_var,
                name=f"origin_{species}_{origin}",
            )
            if penalty_uncovered_origin is not None:
                cost_expr += penalty_uncovered_origin * (1 - c_var)
        if min_coverage_fraction is not None and origins:
            required = min_coverage_fraction * len(origins)
            model.addConstr(
                gp.quicksum(c_vars.values()) >= required,
                name="min_coverage_fraction",
            )
    else:
        for (species, origin), path_ids in origin_to_paths.items():
            if not path_ids:
                print(
                    f"  Warning: origin {origin} of {species} has no candidate paths."
                )
                continue
            model.addConstr(
                gp.quicksum(z[pid] for pid in path_ids) >= 1,
                name=f"origin_{species}_{origin}",
            )

    if species_path_budget:
        for species, cap in species_path_budget.items():
            species_candidates = [c for c in candidates if c.species == species]
            if not species_candidates or cap is None:
                continue
            model.addConstr(
                gp.quicksum(c.cost * z[c.path_id] for c in species_candidates) <= cap,
                name=f"budget_species_{species}",
            )

    model.setObjective(cost_expr, gp.GRB.MINIMIZE)

    start_time = time.time()
    model.optimize()
    elapsed_time = time.time() - start_time
    print_results(model, elapsed_time, phase_label, cost_expr)

    selected_candidates: list[PathCandidate] = []
    selected_by_species: dict[str, list[PathCandidate]] = defaultdict(list)

    if model.SolCount > 0:
        for candidate in candidates:
            if z[candidate.path_id].X > 0.5:
                selected_candidates.append(candidate)
                selected_by_species[candidate.species].append(candidate)

    print(
        f"{phase_label}: selected {len(selected_candidates)} paths "
        f"from {len(candidates)} candidates."
    )

    return {
        "model": model,
        "z": z,
        "cost_expr": cost_expr,
        "selected_candidates": selected_candidates,
        "selected_by_species": selected_by_species,
        "elapsed_time": elapsed_time,
    }


def solve_rehab(
    used_cells_by_species: Mapping[str, set[str]],
    cost_corridor_dict: Mapping[str, float],
    cost_adaptation_dict: Mapping[tuple[str, str], float],
    benefit_adaptation_dict: Mapping[tuple[str, str], float],
    *,
    adjacency: Mapping[str, Sequence[str]],
    all_cells: Sequence[str],
    origin_cells_by_species: Mapping[str, list[str]],
    alpha: float = ALPHA,
    output: bool = True,
    time_limit: float | None = TIME_LIMIT_METHOD,
    heuristics: float = HEURISTICS,
    focus: int = FOCUS,
    phase_label: str = "rehabilitacion_post_paths",
    rehab_budget_by_species: Mapping[str, float] | None = None,
    return_details: bool = False,
) -> dict[str, Any]:
    model = create_model(
        phase_label,
        output=output,
        time_limit_seconds=time_limit,
        heuristics=heuristics,
        focus=focus,
    )

    rehab_vars: dict[tuple[str, str], gp.Var] = {}
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
            rehab_vars[(species, candidate)] = model.addVar(
                vtype=gp.GRB.BINARY, name=f"rehab_{species}_{candidate}"
            )

    if not rehab_vars:
        print("\nNo rehabilitation candidates exist.")
        empty = {sp: set() for sp in SPECIES}
        if return_details:
            return {
                "rehab_selected": empty,
                "model": model,
                "objective_value": None,
                "runtime_seconds": 0.0,
                "elapsed_seconds": 0.0,
                "status": model.Status,
            }
        return empty

    cost_expr = gp.LinExpr()
    benefit_expr = gp.LinExpr()
    for (species, cell), var in rehab_vars.items():
        cost_expr += cost_adaptation_dict[(species, cell)] * var
        benefit_expr += benefit_adaptation_dict[(species, cell)] * var

    model.setObjective(alpha * cost_expr - (1 - alpha) * benefit_expr, gp.GRB.MINIMIZE)

    for cell in all_cells:
        rehab_expr = gp.LinExpr()
        if (species_var := rehab_vars.get(("martes_martes", cell))) is not None:
            rehab_expr += 2 * species_var
        if (species_var := rehab_vars.get(("oryctolagus_cuniculus", cell))) is not None:
            rehab_expr += species_var
        if (species_var := rehab_vars.get(("eliomys_quercinus", cell))) is not None:
            rehab_expr += species_var
        if rehab_expr.size() > 0:
            model.addConstr(rehab_expr <= 2, name=f"compatibility_{cell}")

    used_cells = set(cell for cells in used_cells_by_species.values() for cell in cells)
    used_cost = sum(cost_corridor_dict.get(cell, 0.0) for cell in used_cells)
    print(f"\nCost used in corridors: {used_cost:.2f}")

    if rehab_budget_by_species:
        for species, cap in rehab_budget_by_species.items():
            if cap is None or cap <= 0:
                continue
            species_vars = {
                (sp, cell): var
                for (sp, cell), var in rehab_vars.items()
                if sp == species
            }
            if not species_vars:
                continue
            model.addConstr(
                gp.quicksum(
                    cost_adaptation_dict[(species, cell)] * var
                    for (species, cell), var in species_vars.items()
                )
                <= cap,
                name=f"rehab_budget_{species}",
            )

    start = time.time()
    model.optimize()
    elapsed = time.time() - start
    print_results(model, elapsed, phase_label, cost_expr)

    rehab_selected: dict[str, set[str]] = {sp: set() for sp in SPECIES}
    if model.SolCount > 0:
        for (species, cell), var in rehab_vars.items():
            if var.X > 0.5:
                rehab_selected.setdefault(species, set()).add(cell)

    details = {
        "rehab_selected": rehab_selected,
        "model": model,
        "objective_value": model.ObjVal if model.SolCount > 0 else None,
        "runtime_seconds": model.Runtime,
        "elapsed_seconds": elapsed,
        "status": model.Status,
    }
    return details if return_details else rehab_selected


def run_pipeline() -> None:
    """Complete execution with percentage budgets per species."""
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
    solution_phase1 = solve_path_selection(
        candidates_phase1,
        origin_to_paths_p1,
        phase_label="sin_martes_martes",
        species_path_budget=species_path_budget_phase1,
    )
    selected_by_species_phase1 = solution_phase1.get("selected_by_species", {})
    cost_phase1 = _compute_corridor_cost(selected_by_species_phase1, cost_corridor_dict)
    print(f"Budget used in phase 1 (unique cells): {cost_phase1:.2f}")

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
    solution_martes = solve_path_selection(
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
    rehab_selected_by_species = solve_rehab(
        used_cells_by_species,
        cost_corridor_dict,
        cost_adaptation_dict,
        benefit_adaptation_dict,
        adjacency=adjacency,
        all_cells=all_cells,
        origin_cells_by_species=origin_cells_by_species,
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
        / "graph_model_gurobi_budgeted_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_solution_summary(summary, summary_path)

    maps_dir = root_path / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    map_all = create_solution_map(
        df,
        summary,
        species=None,
        title="Corridors (Gurobi per-species budgets) - All species",
    )
    map_all_path = maps_dir / "graph_model_gurobi_budgeted_map_all.html"
    map_all.save(str(map_all_path))
    print(f"General map saved to {map_all_path}")

    for sp in SPECIES:
        map_sp = create_solution_map(
            df,
            summary,
            species=sp,
            title=f"Corridors (Gurobi per-species budgets) - {sp}",
        )
        map_sp_path = maps_dir / f"graph_model_gurobi_budgeted_map_{sp}.html"
        map_sp.save(str(map_sp_path))
        print(f"Map for {sp} saved to {map_sp_path}")


if __name__ == "__main__":
    run_pipeline()
