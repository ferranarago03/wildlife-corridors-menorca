import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import gurobipy as gp
from gurobipy import GRB

# Add src to path to enable imports when running as script
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from models.gurobi import (
    ALPHA,
    BUDGET,
    FOCUS,
    GAP,
    HEURISTICS,
    MAX_DISTANCE_PIPE,
    SPECIES,
    TIME_LIMIT_PIPE,
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
from models.utils import get_adjacent_cells
from models.visualization import (
    DEFAULT_SPECIES_COLORS,
    SolutionSummary,
    build_solution_summary,
    create_solution_map,
    load_solution_summary,
    save_solution_summary,
)


class GurobiPipeModel:
    """
    Class-based implementation of the 3-phase pipeline using Gurobi.
    Provides structured approach for multi-species corridor optimization.
    """

    def __init__(
        self,
        corridor_cost: dict,
        adaptation_cost: dict,
        adaptation_benefit: dict,
        all_cells: list,
        adjacency: dict,
        origin_cells_by_species: dict,
        species_list: list,
        budget: float = BUDGET,
        alpha: float = ALPHA,
        time_limit_seconds: float = TIME_LIMIT_PIPE,
        gap: float = GAP,
        heuristics: float = HEURISTICS,
        focus: int = FOCUS,
    ):
        # Store all problem data
        self.corridor_cost = corridor_cost
        self.adaptation_cost = adaptation_cost
        self.adaptation_benefit = adaptation_benefit
        self.all_cells = all_cells
        self.adjacency = adjacency
        self.origin_cells_by_species = origin_cells_by_species
        self.full_species_list = species_list
        self.budget = budget
        self.alpha = alpha
        self.time_limit_seconds = time_limit_seconds
        self.gap = gap
        self.heuristics = heuristics
        self.focus = focus

        # Variables dictionaries
        self.x = {}
        self.y = {}
        self.u = {}
        self.rehab = {}

        # Model and cost tracking
        self.model = None
        self.cost_expr = None
        self.final_cost = 0.0
        self.final_status = None

    def _fix_variables(self, y: dict, u: dict, species_list: list):
        """Fix y and u variables to their current solution values"""
        print(f"Fixing variables for {len(species_list)} species...")
        for species in species_list:
            for var_key, var in y.items():
                if var_key[0] == species:
                    var.LB = var.X
                    var.UB = var.X
            for var_key, var in u.items():
                if var_key[0] == species:
                    var.LB = var.X
                    var.UB = var.X

    def _compute_corridor_cost(self, x: dict) -> float:
        """Compute total corridor cost from solution"""
        return sum(
            self.corridor_cost[j] * x[j].X for j in self.all_cells if x[j].X > 0.5
        )

    def _compute_total_cost(self) -> float:
        """Compute total cost including corridors and rehabilitation"""
        corridor_cost = self._compute_corridor_cost(self.x)
        rehab_cost = sum(
            self.adaptation_cost[k] * v.X for k, v in self.rehab.items() if v.X > 0.5
        )
        return corridor_cost + rehab_cost

    def _build_solution_summary(self) -> SolutionSummary:
        """Collect normalized solution data for visualization."""
        built_corridors = {j for j, var in self.x.items() if var.X > 0.5}
        flow_solution = {key: var.X for key, var in self.y.items()}
        rehab_solution = {key: var.X for key, var in self.rehab.items()}

        return build_solution_summary(
            species_list=self.full_species_list,
            origin_cells_by_species=self.origin_cells_by_species,
            built_corridors=built_corridors,
            flow_solution=flow_solution,
            rehab_solution=rehab_solution,
            species_colors=DEFAULT_SPECIES_COLORS,
        )

    def _resolve_summary(self, summary_path: Path | None) -> SolutionSummary | None:
        """
        Load a previously persisted summary if available; otherwise build one from
        the current model and optionally save it for reuse.
        """
        if summary_path is not None:
            try:
                return load_solution_summary(summary_path)
            except FileNotFoundError:
                print(f"No summary found at {summary_path}, rebuilding from model...")
            except Exception as exc:  # Broad catch to avoid blocking map creation
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

    # MAIN PIPELINE METHODS

    def run_phase_1(self):
        """Phase 1: Build corridors for Atelerix algirus"""
        print("\n" + "=" * 60)
        print("PHASE 1: ATELERIX CORRIDORS")
        print("=" * 60)

        self.model = create_model(
            "multi_species_pipe_model",
            time_limit_seconds=self.time_limit_seconds,
            gap=self.gap,
            heuristics=self.heuristics,
            focus=self.focus,
        )
        species_f1 = ["atelerix_algirus"]

        # Create variables
        self.x = create_x_vars(self.model, self.all_cells)
        y_temp = create_y_vars(
            self.model,
            species_f1,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
            max_distance=MAX_DISTANCE_PIPE,
        )
        u_temp = create_u_vars(self.model, species_f1, self.all_cells)

        self.y.update(y_temp)
        self.u.update(u_temp)

        self.model.update()

        # Objective: Minimize corridor cost
        self.cost_expr = gp.LinExpr()
        for j in self.all_cells:
            self.cost_expr += self.corridor_cost[j] * self.x[j]

        self.model.setObjective(self.cost_expr, GRB.MINIMIZE)

        # Constraints
        add_flow_constraints(
            self.model,
            self.y,
            species_f1,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
        )
        add_no_reverse_flow_constraints(
            self.model,
            self.y,
            species_f1,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
        )
        add_activation_constraints(
            self.model,
            self.y,
            self.u,
            species_f1,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
        )

        # Link u and x
        add_link_u_x_single_species(self.model, self.u, self.x, species_f1[0])

        # Budget constraint
        self.model.addConstr(self.cost_expr <= self.budget, name="budget_constraint")

        self.model.update()

        # Solve
        start_time = time.time()
        self.model.optimize()
        elapsed_time = time.time() - start_time

        print_results(self.model, elapsed_time, "Phase 1", self.cost_expr)

        if self.model.SolCount == 0:
            print("Error: Phase 1 found no solution. Aborting.")
            return False

        # Fix variables
        self._fix_variables(self.y, self.u, species_f1)
        return True

    def run_phase_2(self):
        """Phase 2: Extend corridors for remaining species"""
        print("\n" + "=" * 60)
        print("PHASE 2: REMAINING SPECIES CORRIDORS")
        print("=" * 60)

        species_new = [s for s in self.full_species_list if s != "atelerix_algirus"]

        # Create variables for new species
        y_new = create_y_vars(
            self.model,
            species_new,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
            max_distance=MAX_DISTANCE_PIPE,
        )
        u_new = create_u_vars(self.model, species_new, self.all_cells)

        self.y.update(y_new)
        self.u.update(u_new)

        self.model.update()

        # Add constraints for new species
        add_flow_constraints(
            self.model,
            self.y,
            species_new,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
        )
        add_no_reverse_flow_constraints(
            self.model,
            self.y,
            species_new,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
        )
        add_activation_constraints(
            self.model,
            self.y,
            self.u,
            species_new,
            self.origin_cells_by_species,
            self.all_cells,
            self.adjacency,
        )

        # Remove old link constraints from Phase 1
        print("Removing old link constraints from Phase 1...")
        old_constraints = [
            c
            for c in self.model.getConstrs()
            if c.ConstrName.startswith("link_u_x_atelerix_algirus_")
        ]
        for c in old_constraints:
            self.model.remove(c)
        print(f"Removed {len(old_constraints)} old link constraints")

        # Update link constraints for all species
        add_link_u_x_all_species(self.model, self.u, self.x, self.full_species_list)
        add_species_compatibility(
            self.model, self.u, self.all_cells, self.origin_cells_by_species
        )

        self.model.update()

        # Solve
        start_time = time.time()
        self.model.optimize()
        elapsed_time = time.time() - start_time

        print_results(self.model, elapsed_time, "Phase 2", self.cost_expr)

        if self.model.SolCount == 0:
            print("Error: Phase 2 found no solution. Aborting.")
            return False

        # Fix variables
        self._fix_variables(self.y, self.u, species_new)
        return True

    def run_phase_3(self):
        """Phase 3: Add habitat rehabilitation decisions"""
        print("\n" + "=" * 60)
        print("PHASE 3: HABITAT REHABILITATION")
        print("=" * 60)

        # Create rehab variables
        self.rehab = create_rehab_vars(
            self.model,
            self.full_species_list,
            self.all_cells,
            self.origin_cells_by_species,
        )
        self.model.update()

        print(f"Total rehab variables: {len(self.rehab)}")

        # Update objective with adaptation costs and benefits
        benefit = gp.LinExpr()
        for (species, j), var in self.rehab.items():
            self.cost_expr += self.adaptation_cost[(species, j)] * var
            benefit += self.adaptation_benefit[(species, j)] * var

        self.model.setObjective(
            (self.alpha * self.cost_expr) - ((1 - self.alpha) * benefit), GRB.MINIMIZE
        )

        # Remove old budget constraint
        print("Removing old budget constraint...")
        old_budget_constr = self.model.getConstrByName("budget_constraint")
        if old_budget_constr is not None:
            self.model.remove(old_budget_constr)
            print("Old budget constraint removed")

        # Update budget constraint
        self.model.addConstr(
            self.cost_expr <= self.budget, name="budget_constraint_final"
        )

        # Rehab adjacency constraints
        add_rehab_constraints(
            self.model, self.rehab, self.u, self.all_cells, self.adjacency
        )

        self.model.update()

        # Solve
        start_time = time.time()
        self.model.optimize()
        elapsed_time = time.time() - start_time

        self.final_status = self.model.Status

        if self.model.SolCount == 0:
            print("Error: Phase 3 found no solution.")
            print_results(self.model, elapsed_time, "Phase 3", self.cost_expr)
            return False

        # Calculate final cost
        self.final_cost = self._compute_total_cost()

        print_results(self.model, elapsed_time, "Phase 3", self.cost_expr)
        return True

    def visualize_solution(
        self,
        df: gpd.GeoDataFrame,
        save_dir: Path | None = None,
        map_filename: str = "gurobi_pipe_model_class.html",
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
        map_filename_prefix: str = "gurobi_pipe_model_class",
        summary_path: Path | None = None,
    ) -> dict[str, Path]:
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
        saved_paths: dict[str, Path] = {}

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


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run the 3-phase Gurobi pipeline or render maps from a saved summary"
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
        / "gurobi_pipe_summary.json"
    )

    df = gpd.read_parquet(data_path)
    print("Data loaded successfully")

    # Extract origin cells
    origin_cells_by_species = {}
    for species in SPECIES:
        column_name = f"has_{species}"
        species_cells = df[df[column_name]]["grid_id"].tolist()
        if len(species_cells) > 0:
            origin_cells_by_species[species] = species_cells
        else:
            print(f"Warning: No cells found for species {species}")
            origin_cells_by_species[species] = []

    all_cells = df["grid_id"].tolist()
    print(f"\nTotal cells in the grid: {len(all_cells)}")

    # Build cost dictionaries
    cost_corridor_dict = dict(zip(df["grid_id"], df["cost_corridor"]))

    cost_adaptation_dict = {}
    for j in all_cells:
        for species in SPECIES:
            cost_adaptation_dict[(species, j)] = df[df["grid_id"] == j][
                f"cost_adaptation_{species.split('_')[0]}"
            ].values[0]

    benefit_adaptation_dict = {}
    for j in all_cells:
        for species in SPECIES:
            benefit_adaptation_dict[(species, j)] = df[df["grid_id"] == j][
                f"{species.split('_')[1]}_benefit"
            ].values[0]

    # Build adjacency list
    all_cells_set = set(all_cells)
    adjacency = {}
    for cell in all_cells:
        adjacency[cell] = get_adjacent_cells(cell, all_cells_set, df)

    print("\nData processed, starting model pipeline...")

    # Create and run pipeline
    initial_time = time.time()

    pipe_model = GurobiPipeModel(
        corridor_cost=cost_corridor_dict,
        adaptation_cost=cost_adaptation_dict,
        adaptation_benefit=benefit_adaptation_dict,
        all_cells=all_cells,
        adjacency=adjacency,
        origin_cells_by_species=origin_cells_by_species,
        species_list=SPECIES.copy(),
        budget=BUDGET,
        alpha=ALPHA,
        time_limit_seconds=TIME_LIMIT_PIPE,
        gap=GAP,
        heuristics=HEURISTICS,
        focus=FOCUS,
    )

    if args.from_summary:
        pipe_model.visualize_solution(df, summary_path=summary_path)
        pipe_model.visualize_solution_per_species(
            df, summary_path=summary_path, save_dir=Path(".")
        )
        return

    # Run 3-phase pipeline
    if pipe_model.run_phase_1():
        if pipe_model.run_phase_2():
            pipe_model.run_phase_3()

    total_elapsed_time = time.time() - initial_time

    # Visualize and export
    pipe_model.visualize_solution(df, summary_path=summary_path)
    pipe_model.visualize_solution_per_species(df, summary_path=summary_path)

    print(f"\n{'=' * 60}")
    print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
