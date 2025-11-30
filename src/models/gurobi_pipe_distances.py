from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add src to path to enable imports when running as script
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

import geopandas as gpd
import gurobipy as gp
from gurobipy import GRB

from models.gurobi import (
    ALPHA,
    BUDGET,
    FOCUS,
    GAP,
    HEURISTICS,
    MAX_DISTANCE_METHOD,
    MAX_MANHATTAN_METHOD,
    SPECIES,
    TIME_LIMIT_METHOD,
    add_activation_constraints,
    add_flow_constraints,
    add_link_u_x_all_species,
    add_link_u_x_single_species,
    add_no_reverse_flow_constraints,
    add_rehab_constraints,
    add_species_compatibility,
    create_model,
    create_rehab_vars,
    create_u_vars,
    create_x_vars,
    create_y_vars,
    print_results,
)
from models.utils import get_adjacent_cells, manhattan_distance
from models.visualization import (
    DEFAULT_SPECIES_COLORS,
    SolutionSummary,
    build_solution_summary,
    create_solution_map,
    load_solution_summary,
    save_solution_summary,
)

# -------------------- Methodology Model Class -------------------- #


class GurobiPipeMethodologyModel:
    def __init__(
        self,
        corridor_cost: Dict[str, float],
        adaptation_cost: Dict[Tuple[str, str], float],
        adaptation_benefit: Dict[Tuple[str, str], float],
        all_cells: List[str],
        adjacency: Dict[str, List[str]],
        origin_cells_by_species: Dict[str, List[str]],
        species_list: List[str],
        budget: float = BUDGET,
        alpha: float = ALPHA,
        time_limit_seconds: float = TIME_LIMIT_METHOD,
        max_distance: int = MAX_DISTANCE_METHOD,
        gap: float = GAP,
        heuristics: float = HEURISTICS,
        focus: int = FOCUS,
    ):
        self.corridor_cost = corridor_cost
        self.adaptation_cost = adaptation_cost
        self.adaptation_benefit = adaptation_benefit
        self.all_cells = all_cells
        self.adjacency = adjacency
        self.origin_cells_by_species = origin_cells_by_species
        self.species_list = species_list
        self.budget = budget
        self.alpha = alpha
        self.time_limit_seconds = time_limit_seconds
        self.gap = gap
        self.heuristics = heuristics
        self.focus = focus
        self.max_manhattan = MAX_MANHATTAN_METHOD
        self.max_distance = max_distance

        # Containers for results of step 1
        self.individual_corridors: Dict[str, Set[str]] = {}

        # Combined model variables
        self.model: gp.Model | None = None
        self.x: Dict[str, gp.Var] = {}
        self.y: Dict[Tuple[str, str, str, str], gp.Var] = {}
        self.u: Dict[Tuple[str, str], gp.Var] = {}
        self.rehab: Dict[Tuple[str, str], gp.Var] = {}
        self.cost_expr: gp.LinExpr | None = None

    # --------- Step 1: Individual species runs --------- #

    def run_individual_species(
        self, parallel: bool = True, threads_per_model: int | None = None
    ):
        print(
            "\n================ STEP 1: Individual species corridor extraction ================"
        )

        def _solve_for_species(species: str, threads: int | None):
            print(f"\n-- Running individual model for species: {species}")
            model = create_model(
                f"ind_{species}",
                time_limit_seconds=self.time_limit_seconds,
                gap=self.gap,
                heuristics=self.heuristics,
                focus=self.focus,
            )
            if threads is not None:
                model.setParam("Threads", threads)
            x = create_x_vars(model, self.all_cells)
            y = create_y_vars(
                model,
                [species],
                self.origin_cells_by_species,
                self.all_cells,
                self.adjacency,
                max_distance=self.max_distance,
            )
            u = create_u_vars(model, [species], self.all_cells)
            model.update()
            cost_expr = gp.LinExpr()
            for j in self.all_cells:
                cost_expr += self.corridor_cost[j] * x[j]
            model.setObjective(cost_expr, GRB.MINIMIZE)
            # Flow + activation
            add_flow_constraints(
                model,
                y,
                [species],
                self.origin_cells_by_species,
                self.all_cells,
                self.adjacency,
            )
            add_activation_constraints(
                model,
                y,
                u,
                [species],
                self.origin_cells_by_species,
                self.all_cells,
                self.adjacency,
            )
            # Link u-x
            add_link_u_x_single_species(model, u, x, species)
            # Budget
            model.addConstr(cost_expr <= self.budget, name=f"budget_{species}")
            model.update()
            start = time.time()
            model.optimize()
            elapsed = time.time() - start
            print_results(model, elapsed, f"Individual-{species}", cost_expr)
            corridors = set()
            if model.SolCount > 0:
                corridors = {j for j in self.all_cells if x[j].X > 0.5}
                print(f"Corridors built for {species}: {len(corridors)}")
            else:
                print(f"No feasible solution for species {species}.")
            return species, corridors

        if parallel:
            if threads_per_model is None:
                cpu = os.cpu_count() or 4
                threads_per_model = max(1, cpu // max(1, len(self.species_list)))
            print(f"Parallel mode ON. Threads per model: {threads_per_model}")
            with ThreadPoolExecutor(max_workers=len(self.species_list)) as ex:
                futures = [
                    ex.submit(_solve_for_species, sp, threads_per_model)
                    for sp in self.species_list
                ]
                for fut in as_completed(futures):
                    sp, corr = fut.result()
                    self.individual_corridors[sp] = corr
        else:
            print("Parallel mode OFF. Running sequentially.")
            for species in self.species_list:
                sp, corr = _solve_for_species(species, threads_per_model)
                self.individual_corridors[sp] = corr

    # --------- Step 2: Prune distant variables --------- #

    def prune_far_variables(self):
        print(
            f"\n================ STEP 2: Pruning distant variables (> Manhattan {self.max_manhattan}) ================"
        )
        # Precompute nearest distance per species per cell
        self.far_cells_by_species: Dict[str, Set[str]] = {}
        for species, corridors in self.individual_corridors.items():
            far_set = set()
            if not corridors:
                # If no corridors, mark all cells as far to effectively disable species
                far_set = set(self.all_cells)
            else:
                for j in self.all_cells:
                    min_dist = min(manhattan_distance(j, c) for c in corridors)
                    if min_dist > self.max_manhattan:
                        far_set.add(j)
            self.far_cells_by_species[species] = far_set
            print(f"Species {species}: far cells count = {len(far_set)}")

    # --------- Step 3: Combined multi-species model (without rehab) --------- #

    def run_combined_model(self):
        print(
            "\n================ STEP 3: Combined multi-species optimization ================"
        )
        self.model = create_model(
            "combined_multi_species",
            time_limit_seconds=self.time_limit_seconds,
            gap=self.gap,
            heuristics=self.heuristics,
            focus=self.focus,
        )
        self.x = create_x_vars(self.model, self.all_cells)
        self.y = create_y_vars(
            self.model,
            self.species_list,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
            max_distance=self.max_distance,
            far_cells_by_species=getattr(self, "far_cells_by_species", {}),
        )
        self.u = create_u_vars(self.model, self.species_list, self.all_cells)
        self.model.update()

        # Objective: corridor cost only at this stage
        self.cost_expr = gp.LinExpr()
        for j in self.all_cells:
            self.cost_expr += self.corridor_cost[j] * self.x[j]
        self.model.setObjective(self.cost_expr, GRB.MINIMIZE)

        # Constraints
        add_flow_constraints(
            self.model,
            self.y,
            self.species_list,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
        )
        add_no_reverse_flow_constraints(
            self.model,
            self.y,
            self.species_list,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
        )
        add_activation_constraints(
            self.model,
            self.y,
            self.u,
            self.species_list,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
        )

        # Link constraints aggregated
        add_link_u_x_all_species(self.model, self.u, self.x, self.species_list)

        # Species compatibility
        add_species_compatibility(
            self.model, self.u, self.all_cells, self.origin_cells_by_species
        )

        # Budget
        self.model.addConstr(self.cost_expr <= self.budget, name="budget_combined")

        # Variables already pruned during creation
        print("Variables for far cells were not created (pruned at creation time)")

        self.model.update()
        start = time.time()
        self.model.optimize()
        elapsed = time.time() - start
        print_results(self.model, elapsed, "Combined", self.cost_expr)
        return self.model.SolCount > 0

    # --------- Step 4: Rehabilitation phase --------- #

    def run_rehabilitation(self):
        print("\n================ STEP 4: Rehabilitation optimization ================")
        if self.model is None or self.model.SolCount == 0:
            print("Cannot run rehabilitation phase: no combined solution.")
            return False
        # Fix corridor usage variables to current solution
        for (species, j), var in self.u.items():
            var.LB = var.X
            var.UB = var.X
        for key, var in self.y.items():
            var.LB = var.X
            var.UB = var.X

        # Create rehab vars
        self.rehab = create_rehab_vars(
            self.model, self.species_list, self.all_cells, self.origin_cells_by_species
        )
        self.model.update()

        benefit = gp.LinExpr()
        for (species, j), var in self.rehab.items():
            self.cost_expr += self.adaptation_cost[(species, j)] * var
            benefit += self.adaptation_benefit[(species, j)] * var

        # Remove previous budget if exists
        old_budget = self.model.getConstrByName("budget_combined")
        if old_budget is not None:
            self.model.remove(old_budget)

        # New objective with alpha weighting cost-benefit
        self.model.setObjective(
            (self.alpha * self.cost_expr) - ((1 - self.alpha) * benefit), GRB.MINIMIZE
        )
        self.model.addConstr(self.cost_expr <= self.budget, name="budget_final")

        # Rehab constraints
        add_rehab_constraints(
            self.model, self.rehab, self.u, self.all_cells, self.adjacency
        )
        self.model.update()
        start = time.time()
        self.model.optimize()
        elapsed = time.time() - start
        print_results(self.model, elapsed, "Rehabilitation", self.cost_expr)
        return (
            self.model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]
            and self.model.SolCount > 0
        )

    def _build_solution_summary(self) -> SolutionSummary:
        """Collect normalized solution data for visualization."""
        built_corridors = {j for j, var in self.x.items() if var.X > 0.5}
        flow_solution = {key: var.X for key, var in self.y.items()}
        rehab_solution = {key: var.X for key, var in self.rehab.items()}

        return build_solution_summary(
            species_list=self.species_list,
            origin_cells_by_species=self.origin_cells_by_species,
            built_corridors=built_corridors,
            flow_solution=flow_solution,
            rehab_solution=rehab_solution,
            species_colors=DEFAULT_SPECIES_COLORS,
        )

    def _resolve_summary(self, summary_path: Path | None) -> SolutionSummary | None:
        """
        Load a saved summary if it exists; otherwise build one from the current
        model and optionally persist it.
        """
        if summary_path is not None:
            try:
                return load_solution_summary(summary_path)
            except FileNotFoundError:
                print(f"No summary found at {summary_path}, rebuilding from model...")
            except Exception as exc:
                print(
                    f"Warning: could not load summary from {summary_path} ({exc}). "
                    "Rebuilding from model instead."
                )

        if self.model is None or self.model.SolCount == 0:
            print("No solution to visualize.")
            return None

        summary = self._build_solution_summary()
        if summary_path is not None:
            save_solution_summary(summary, summary_path)
        return summary

    def export_solution_summary(self, summary_path: Path) -> Path:
        """Public helper to write the current solution summary to disk."""
        summary = self._build_solution_summary()
        return save_solution_summary(summary, summary_path)

    # --------- Visualization & Export --------- #

    def visualize_solution(
        self,
        df: gpd.GeoDataFrame,
        save_dir: Path | None = None,
        map_filename: str = "gurobi_pipe_distances.html",
        summary_path: Path | None = None,
    ):
        """Create interactive map visualization of the solution"""
        # Default to maps directory in project root
        if save_dir is None:
            save_dir = Path(__file__).parent.parent.parent / "maps"
        summary = self._resolve_summary(summary_path)
        if summary is None:
            return

        # Create save directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("CORRIDORS USED PER SPECIES")
        print(f"{'=' * 60}")
        for species in summary.species_list:
            origin_cells = self.origin_cells_by_species.get(species, [])
            corridors_used = summary.corridors_by_species.get(species, set())
            rehab_cells = summary.rehabilitated_by_species.get(species, set())
            print(f"\n{species}:")
            print(f"  Origins: {len(origin_cells)}")
            print(f"  Corridors used: {len(corridors_used)}")
            print(f"  Rehabilitated cells: {len(rehab_cells)}")

        print(f"\nTotal built corridors: {len(summary.built_corridors)}")

        folium_map = create_solution_map(
            df=df,
            summary=summary,
            species=None,
            title="Legend - Gurobi Model",
        )

        save_path = save_dir / map_filename
        folium_map.save(save_path)
        print(f"\n✓ Map saved to: {save_path}")

    def visualize_solution_per_species(
        self,
        df: gpd.GeoDataFrame,
        save_dir: Path | None = None,
        map_filename_prefix: str = "gurobi_pipe_distances",
        summary_path: Path | None = None,
    ) -> Dict[str, Path]:
        """
        Create one map per species highlighting its corridors and adaptations.
        Returns a mapping of species to saved file paths.
        """
        # Default to maps directory in project root
        if save_dir is None:
            save_dir = Path(__file__).parent.parent.parent / "maps"
        summary = self._resolve_summary(summary_path)
        if summary is None:
            return {}
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: Dict[str, Path] = {}

        for species in summary.species_list:
            folium_map = create_solution_map(
                df=df,
                summary=summary,
                species=species,
                title=f"Legend - {species}",
            )
            filename = f"{map_filename_prefix}_{species}.html"
            save_path = save_dir / filename
            folium_map.save(save_path)
            saved_paths[species] = save_path

        print("\n✓ Species-specific maps saved:")
        for species, path in saved_paths.items():
            print(f"  - {species}: {path}")
        return saved_paths


# -------------------- Runner -------------------- #


def main():
    parser = argparse.ArgumentParser(description="Run Gurobi multi-species methodology")
    parser.add_argument(
        "--parallel",
        dest="parallel",
        action="store_true",
        default=True,
        help="Run individual species in parallel (default: True)",
    )
    parser.add_argument(
        "--serial",
        dest="parallel",
        action="store_false",
        help="Run individual species sequentially",
    )
    parser.add_argument(
        "--threads-per-model",
        dest="threads_per_model",
        type=int,
        default=3,
        help="Threads per individual model in parallel mode",
    )
    parser.add_argument(
        "--max-manhattan",
        dest="max_manhattan",
        type=int,
        default=MAX_MANHATTAN_METHOD,
        help="Max Manhattan distance for pruning",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Path to store/load the solution summary JSON used by map rendering",
    )
    parser.add_argument(
        "--from-summary",
        action="store_true",
        help="Skip solving and only render maps from an existing summary",
    )
    args = parser.parse_args()

    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "processed_dataset.parquet"
    summary_path = (
        args.summary_path
        if args.summary_path is not None
        else root_path
        / "data"
        / "experiments"
        / "summaries"
        / "gurobi_method_summary.json"
    )
    df = gpd.read_parquet(data_path)
    print("Data loaded")
    origin_cells_by_species = {}
    for species in SPECIES:
        col = f"has_{species}"
        cells = df[df[col]]["grid_id"].tolist()
        origin_cells_by_species[species] = cells
        if not cells:
            print(f"Warning: no origin cells for {species}")
    all_cells = df["grid_id"].tolist()
    cost_corridor = dict(zip(df["grid_id"], df["cost_corridor"]))
    cost_adapt = {}
    benefit_adapt = {}
    for j in all_cells:
        for species in SPECIES:
            row_slice = df[df["grid_id"] == j]
            cost_adapt[(species, j)] = row_slice[
                f"cost_adaptation_{species.split('_')[0]}"
            ].values[0]
            benefit_adapt[(species, j)] = row_slice[
                f"{species.split('_')[1]}_benefit"
            ].values[0]
    all_cells_set = set(all_cells)
    adjacency = {
        cell: get_adjacent_cells(cell, all_cells_set, df) for cell in all_cells
    }
    model = GurobiPipeMethodologyModel(
        corridor_cost=cost_corridor,
        adaptation_cost=cost_adapt,
        adaptation_benefit=benefit_adapt,
        all_cells=all_cells,
        adjacency=adjacency,
        origin_cells_by_species=origin_cells_by_species,
        species_list=SPECIES.copy(),
        budget=BUDGET,
        alpha=ALPHA,
        time_limit_seconds=TIME_LIMIT_METHOD,
        gap=GAP,
        heuristics=HEURISTICS,
        focus=FOCUS,
    )
    model.max_manhattan = args.max_manhattan

    if args.from_summary:
        model.visualize_solution(df, summary_path=summary_path)
        model.visualize_solution_per_species(df, summary_path=summary_path)
        return

    # Execute methodology steps

    init_time = time.time()

    model.run_individual_species(
        parallel=args.parallel, threads_per_model=args.threads_per_model
    )
    model.prune_far_variables()
    if model.run_combined_model():
        model.run_rehabilitation()

    total_elapsed_time = time.time() - init_time

    model.visualize_solution(df, summary_path=summary_path)
    model.visualize_solution_per_species(df, summary_path=summary_path)

    print(f"\n{'=' * 60}")
    print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
