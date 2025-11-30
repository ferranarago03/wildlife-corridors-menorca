from __future__ import annotations

import time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import geopandas as gpd
import ortools.linear_solver.pywraplp as pywraplp
import ortools.sat.python.cp_model as cp_model

from ..visualization import (
    DEFAULT_SPECIES_COLORS,
    SolutionSummary,
    build_solution_summary,
    create_solution_map,
)
from .constants import ALPHA, BUDGET, NUM_THREADS, TIME_LIMIT_MS
from .reporting import print_results_cpsat


class CPSATPipeModel:
    """
    Pipeline implementation using the CP-SAT solver.
    Mirrors the structure of the LP implementation but uses cp_model API.
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
        time_limit_ms: float = TIME_LIMIT_MS,
        num_threads: int = NUM_THREADS,
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
        self.time_limit_ms = time_limit_ms
        self.num_threads = num_threads

        # Dictionaries to store variable solutions
        self.sol_x = {}
        self.sol_y = {}
        self.sol_u = {}
        self.sol_rehab = {}

        self.final_cost = 0.0
        self.final_model_status = None

        # Pre-compute scaling information for CP-SAT integer requirements
        self.cost_scaling_factor = self._determine_scaling_factor()
        self.alpha_fraction = Fraction(self.alpha).limit_denominator(10_000)
        self.budget_scaled = self._scale_value(self.budget)

    def _determine_scaling_factor(self) -> int:
        """Determine the minimal power-of-ten scaling factor for all costs."""
        values = list(self.corridor_cost.values())
        values.extend(self.adaptation_cost.values())
        values.extend(self.adaptation_benefit.values())

        max_decimal_places = 0
        for value in values:
            if value is None:
                continue
            dec_value = Decimal(str(value)).normalize()
            if dec_value.is_nan():
                continue
            if dec_value == 0:
                continue
            exponent = dec_value.as_tuple().exponent
            if exponent < 0:
                max_decimal_places = max(max_decimal_places, -exponent)

        return max(1, 10 ** min(max_decimal_places, 6))

    def _scale_value(self, value: float) -> int:
        """Scale a float to an integer using the common scaling factor."""
        return int(round(value * self.cost_scaling_factor))

    @staticmethod
    def _is_feasible_status(status: int) -> bool:
        return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def _compute_corridor_cost(self, x_solution: dict) -> float:
        return sum(self.corridor_cost[j] * x_solution.get(j, 0) for j in self.all_cells)

    def _compute_total_cost(self) -> float:
        corridor_cost = self._compute_corridor_cost(self.sol_x)
        rehab_cost = sum(self.adaptation_cost[k] * v for k, v in self.sol_rehab.items())
        return corridor_cost + rehab_cost

    def _create_solver(self) -> cp_model.CpSolver:
        return cp_model.CpSolver()

    def set_solver_parameters(
        self,
        solver: cp_model.CpSolver,
        num_threads: int,
        time_limit_ms: int,
        gap: float = 0.2,
        log_search_progress: bool = True,
    ) -> None:
        """Configure CP-SAT solver parameters with sensible defaults."""
        params = solver.parameters

        if num_threads and num_threads > 0:
            params.num_search_workers = max(1, num_threads)

        if time_limit_ms and time_limit_ms > 0:
            params.max_time_in_seconds = time_limit_ms / 1000.0

        if gap is not None and gap >= 0:
            params.relative_gap_limit = gap

        params.log_search_progress = log_search_progress
        params.cp_model_presolve = True
        params.linearization_level = 2

    # --- REUSABLE MODEL BUILDING METHODS (CP-SAT) ---

    def _create_x_vars(self, model: cp_model.CpModel):
        """Creates x variables (corridor built) for all cells."""
        x = {}
        for j in self.all_cells:
            x[j] = model.NewBoolVar(f"x_{j}")
        return x

    def _create_y_vars(self, model: cp_model.CpModel, species_list: list):
        """Creates y variables (flow) for a list of species."""
        y = {}
        for species in species_list:
            origin_cells = self.origin_cells_by_species[species]
            for r in origin_cells:
                for j in self.all_cells:
                    for k in self.adjacency[j]:
                        y[(species, r, j, k)] = model.NewBoolVar(
                            f"y_{species}_{r}_{j}_{k}"
                        )
        return y

    def _create_u_vars(self, model: cp_model.CpModel, species_list: list):
        """Creates u variables (cell used) for a list of species."""
        u = {}
        for species in species_list:
            for j in self.all_cells:
                u[(species, j)] = model.NewBoolVar(f"u_{species}_{j}")
        return u

    def _create_rehab_vars(self, model: cp_model.CpModel, species_list: list):
        """Creates rehab variables (rehabilitation) for a list of species."""
        rehab = {}
        for species in species_list:
            origin_cells = self.origin_cells_by_species[species]
            for j in self.all_cells:
                if j not in origin_cells:
                    rehab[(species, j)] = model.NewBoolVar(f"rehab_{species}_{j}")
        return rehab

    def _add_flow_constraints(
        self, model: cp_model.CpModel, y: dict, species_list: list
    ):
        """Adds outflow (1) and flow conservation (2) constraints."""
        print(f"Adding {len(species_list)} set(s) of flow constraints...")
        for species in species_list:
            origin_cells = self.origin_cells_by_species[species]
            # 1. Outflow
            for r in origin_cells:
                model.Add(
                    sum(
                        y[(species, r, r, j)]
                        for j in self.adjacency[r]
                        if (species, r, r, j) in y
                    )
                    >= 1
                ).WithName(f"origin_outflow_{species}_{r}")
            # 2. Flow conservation
            for j in self.all_cells:
                for r in origin_cells:
                    if j in origin_cells:
                        continue

                    inflow = sum(
                        y[(species, r, i, j)]
                        for i in self.adjacency[j]
                        if (species, r, i, j) in y
                    )
                    outflow = sum(
                        y[(species, r, j, k)]
                        for k in self.adjacency[j]
                        if k != r and (species, r, j, k) in y
                    )
                    model.Add(inflow - outflow == 0).WithName(
                        f"flow_cons_{species}_{r}_{j}"
                    )

    def _add_no_reverse_flow_constraints(
        self, model: cp_model.CpModel, y: dict, species_list: list
    ):
        """Adds no-reverse-flow constraints."""
        print(f"Adding {len(species_list)} set(s) of no-reverse-flow constraints...")
        for species in species_list:
            origin_cells = self.origin_cells_by_species[species]
            for r in origin_cells:
                for j in self.all_cells:
                    for k in self.adjacency[j]:
                        if j < k:
                            if (species, r, k, j) in y and (species, r, j, k) in y:
                                model.Add(
                                    y[(species, r, j, k)] + y[(species, r, k, j)] <= 1
                                ).WithName(f"no_reverse_{species}_{r}_{j}_{k}")

    def _add_activation_constraints(
        self, model: cp_model.CpModel, y: dict, u: dict, species_list: list
    ):
        """Adds cell activation constraints (3) (link y -> u)."""
        print(f"Adding {len(species_list)} set(s) of activation constraints...")
        for species in species_list:
            origin_cells = self.origin_cells_by_species[species]
            for j in self.all_cells:
                inflow_sum = sum(
                    y[(species, r, i, j)]
                    for r in origin_cells
                    for i in self.adjacency[j]
                    if (species, r, i, j) in y
                )
                outflow_sum = sum(
                    y[(species, r, j, k)]
                    for r in origin_cells
                    for k in self.adjacency[j]
                    if (species, r, j, k) in y
                )
                usage_sum = inflow_sum + outflow_sum

                model.Add(usage_sum >= 1).OnlyEnforceIf(u[(species, j)])
                model.Add(usage_sum == 0).OnlyEnforceIf(u[(species, j)].Not())

    def _add_fix_y_constraints(self, model: cp_model.CpModel, y: dict, sol_y: dict):
        """Adds constraints to fix Y variables to their previous solution."""
        print(f"Fixing {len(sol_y)} Y variables...")
        for k, v_sol in sol_y.items():
            if k in y:
                model.Add(y[k] == int(round(v_sol))).WithName(
                    f"fix_y_{k[0]}_{k[1]}_{k[2]}_{k[3]}"
                )

    def _add_fix_u_constraints(self, model: cp_model.CpModel, u: dict, sol_u: dict):
        """Adds constraints to fix U variables to their previous solution."""
        print(f"Fixing {len(sol_u)} U variables...")
        for k, v_sol in sol_u.items():
            if k in u:
                model.Add(u[k] == int(round(v_sol))).WithName(f"fix_u_{k[0]}_{k[1]}")

    # --- MAIN PIPELINE METHODS (CP-SAT) ---

    def run_phase_1(self) -> bool:
        print("\n--- Starting Phase 1: Atelerix Corridors (CP-SAT) ---")
        model = cp_model.CpModel()
        species_f1 = ["atelerix_algirus"]

        # --- Phase 1 Variables ---
        x = self._create_x_vars(model)
        y = self._create_y_vars(model, species_f1)
        u = self._create_u_vars(model, species_f1)

        # --- Phase 1 Objective ---
        cost_expr = sum(
            self._scale_value(self.corridor_cost[j]) * x[j] for j in self.all_cells
        )
        model.Minimize(cost_expr)

        # --- Phase 1 Constraints ---
        self._add_flow_constraints(model, y, species_f1)
        self._add_no_reverse_flow_constraints(model, y, species_f1)
        self._add_activation_constraints(model, y, u, species_f1)

        for j in self.all_cells:
            model.Add(u[(species_f1[0], j)] <= x[j]).WithName(
                f"link_u_x_{species_f1[0]}_{j}"
            )

        model.Add(cost_expr <= self.budget_scaled).WithName("budget_constraint")

        solver = self._create_solver()
        self.set_solver_parameters(
            solver,
            self.num_threads,
            self.time_limit_ms,
            gap=0.2,
        )
        start_time = time.time()
        status = solver.Solve(model)
        elapsed_time = time.time() - start_time

        if not self._is_feasible_status(status):
            print("Error: Phase 1 found no solution. Aborting.")
            print_results_cpsat(
                solver,
                status,
                elapsed_time,
                "Phase 1",
            )
            return False

        self.sol_x = {j: solver.Value(v) for j, v in x.items()}
        self.sol_y = {k: solver.Value(v) for k, v in y.items()}
        self.sol_u = {k: solver.Value(v) for k, v in u.items()}

        phase_cost = self._compute_corridor_cost(self.sol_x)
        num_vars = len(x) + len(y) + len(u)

        print_results_cpsat(
            solver,
            status,
            elapsed_time,
            "Phase 1",
            actual_cost=phase_cost,
            objective_scale=float(self.cost_scaling_factor),
            num_variables=num_vars,
        )
        return True

    def run_phase_2(self) -> bool:
        print("\n--- Starting Phase 2: Remaining Corridors (CP-SAT) ---")
        model = cp_model.CpModel()
        species_new = [s for s in self.full_species_list if s != "atelerix_algirus"]

        x = self._create_x_vars(model)
        y = self._create_y_vars(model, self.full_species_list)
        u = self._create_u_vars(model, self.full_species_list)

        cost_expr = sum(
            self._scale_value(self.corridor_cost[j]) * x[j] for j in self.all_cells
        )
        model.Minimize(cost_expr)

        self._add_fix_y_constraints(model, y, self.sol_y)
        self._add_fix_u_constraints(model, u, self.sol_u)

        self._add_flow_constraints(model, y, species_new)
        self._add_no_reverse_flow_constraints(model, y, species_new)
        self._add_activation_constraints(model, y, u, species_new)

        for j in self.all_cells:
            lhs = sum(u[(species, j)] for species in self.full_species_list)
            model.Add(lhs <= len(self.full_species_list) * x[j]).WithName(
                f"link_u_x_all_{j}"
            )

        model.Add(cost_expr <= self.budget_scaled).WithName("budget_constraint")

        martes_origin_cells = self.origin_cells_by_species.get("martes_martes", [])
        oryctolagus_origin_cells = self.origin_cells_by_species.get(
            "oryctolagus_cuniculus", []
        )
        eliomys_origin_cells = self.origin_cells_by_species.get("eliomys_quercinus", [])

        for j in self.all_cells:
            if j in martes_origin_cells and (
                j in oryctolagus_origin_cells or j in eliomys_origin_cells
            ):
                continue
            term_martes = u.get(("martes_martes", j), 0)
            term_oryctolagus = u.get(("oryctolagus_cuniculus", j), 0)
            term_eliomys = u.get(("eliomys_quercinus", j), 0)
            model.Add(2 * term_martes + term_oryctolagus + term_eliomys <= 2).WithName(
                f"species_compatibility_{j}"
            )

        solver = self._create_solver()
        self.set_solver_parameters(
            solver,
            self.num_threads,
            self.time_limit_ms,
            gap=0.2,
        )
        start_time = time.time()
        status = solver.Solve(model)
        elapsed_time = time.time() - start_time

        if not self._is_feasible_status(status):
            print("Error: Phase 2 found no solution. Aborting.")
            print_results_cpsat(
                solver,
                status,
                elapsed_time,
                "Phase 2",
            )
            return False

        self.sol_x = {j: solver.Value(v) for j, v in x.items()}
        self.sol_y = {k: solver.Value(v) for k, v in y.items()}
        self.sol_u = {k: solver.Value(v) for k, v in u.items()}

        phase_cost = self._compute_corridor_cost(self.sol_x)
        num_vars = len(x) + len(y) + len(u)

        print_results_cpsat(
            solver,
            status,
            elapsed_time,
            "Phase 2",
            actual_cost=phase_cost,
            objective_scale=float(self.cost_scaling_factor),
            num_variables=num_vars,
        )
        return True

    def run_phase_3(self) -> bool:
        print("\n--- Starting Phase 3: Habitat Rehabilitation (CP-SAT) ---")
        model = cp_model.CpModel()
        self.sol_rehab = {}

        x = self._create_x_vars(model)
        y = self._create_y_vars(model, self.full_species_list)
        u = self._create_u_vars(model, self.full_species_list)
        rehab = self._create_rehab_vars(model, self.full_species_list)

        cost_corridor_int = sum(
            self._scale_value(self.corridor_cost[j]) * x[j] for j in self.all_cells
        )
        cost_adaptation_int = sum(
            self._scale_value(self.adaptation_cost[k]) * v for k, v in rehab.items()
        )
        benefit_adaptation_int = sum(
            self._scale_value(self.adaptation_benefit[k]) * v for k, v in rehab.items()
        )

        total_cost_expr_int = cost_corridor_int + cost_adaptation_int

        numerator = self.alpha_fraction.numerator
        denominator = self.alpha_fraction.denominator

        model.Minimize(
            (numerator * total_cost_expr_int)
            - ((denominator - numerator) * benefit_adaptation_int)
        )

        self._add_fix_y_constraints(model, y, self.sol_y)
        self._add_fix_u_constraints(model, u, self.sol_u)

        for j in self.all_cells:
            lhs = sum(u[(species, j)] for species in self.full_species_list)
            model.Add(lhs <= len(self.full_species_list) * x[j]).WithName(
                f"link_u_x_all_{j}"
            )

        martes_origin_cells = self.origin_cells_by_species.get("martes_martes", [])
        oryctolagus_origin_cells = self.origin_cells_by_species.get(
            "oryctolagus_cuniculus", []
        )
        eliomys_origin_cells = self.origin_cells_by_species.get("eliomys_quercinus", [])
        for j in self.all_cells:
            if j in martes_origin_cells and (
                j in oryctolagus_origin_cells or j in eliomys_origin_cells
            ):
                continue
            term_martes = u.get(("martes_martes", j), 0)
            term_oryctolagus = u.get(("oryctolagus_cuniculus", j), 0)
            term_eliomys = u.get(("eliomys_quercinus", j), 0)
            model.Add(2 * term_martes + term_oryctolagus + term_eliomys <= 2).WithName(
                f"species_compatibility_u_{j}"
            )

        model.Add(total_cost_expr_int <= self.budget_scaled).WithName(
            "budget_constraint_final"
        )

        for (species, j), var in rehab.items():
            adj_usage = sum(
                int(round(self.sol_u.get((species, k), 0))) for k in self.adjacency[j]
            )
            model.Add(var <= adj_usage).WithName(f"rehab_adjacent_{species}_{j}")

        for j in self.all_cells:
            term_martes = rehab.get(("martes_martes", j), 0)
            term_oryctolagus = rehab.get(("oryctolagus_cuniculus", j), 0)
            term_eliomys = rehab.get(("eliomys_quercinus", j), 0)
            model.Add(2 * term_martes + term_oryctolagus + term_eliomys <= 2).WithName(
                f"rehab_compatibility_{j}"
            )

        solver = self._create_solver()
        self.set_solver_parameters(
            solver,
            self.num_threads,
            self.time_limit_ms,
            gap=0.2,
        )
        start_time = time.time()
        status = solver.Solve(model)
        elapsed_time = time.time() - start_time
        self.final_model_status = status

        objective_scale = float(self.cost_scaling_factor * denominator)

        if not self._is_feasible_status(status):
            print("Error: Phase 3 found no solution.")
            print_results_cpsat(
                solver,
                status,
                elapsed_time,
                "Phase 3",
                objective_scale=objective_scale,
            )
            return False

        self.sol_x = {j: solver.Value(v) for j, v in x.items()}
        self.sol_y = {k: solver.Value(v) for k, v in y.items()}
        self.sol_u = {k: solver.Value(v) for k, v in u.items()}
        self.sol_rehab = {k: solver.Value(v) for k, v in rehab.items()}

        self.final_cost = self._compute_total_cost()
        num_vars = len(x) + len(y) + len(u) + len(rehab)

        print_results_cpsat(
            solver,
            status,
            elapsed_time,
            "Phase 3",
            actual_cost=self.final_cost,
            objective_scale=objective_scale,
            num_variables=num_vars,
        )
        return True

    def _build_solution_summary(self) -> SolutionSummary:
        """Collect normalized solution data for visualization."""
        built_corridors = {j for j, val in self.sol_x.items() if val > 0.5}
        return build_solution_summary(
            species_list=self.full_species_list,
            origin_cells_by_species=self.origin_cells_by_species,
            built_corridors=built_corridors,
            flow_solution=self.sol_y,
            rehab_solution=self.sol_rehab,
            species_colors=DEFAULT_SPECIES_COLORS,
        )

    def visualize_solution(
        self,
        df: gpd.GeoDataFrame,
        save_dir: Path | None = None,
        map_filename: str = "ortools_pipe_cpsat.html",
    ) -> None:
        # Default to maps directory in project root
        if save_dir is None:
            save_dir = Path(__file__).parent.parent.parent.parent / "maps"
        # Create save directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.final_model_status not in [
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
        ]:
            print("No solution to visualize.")
            return

        summary = self._build_solution_summary()

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
            title="Legend - OR-Tools CP-SAT",
        )

        save_path = save_dir / map_filename
        folium_map.save(save_path)
        print(f"\n✓ Map saved to: {save_path}")

    def visualize_solution_per_species(
        self,
        df: gpd.GeoDataFrame,
        save_dir: Path | None = None,
        map_filename_prefix: str = "ortools_pipe_cpsat",
    ) -> dict[str, Path]:
        """
        Create one map per species highlighting its corridors and adaptations.
        Returns a mapping of species to saved file paths.
        """
        # Default to maps directory in project root
        if save_dir is None:
            save_dir = Path(__file__).parent.parent.parent.parent / "maps"
        if self.final_model_status not in [
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
        ]:
            print("No solution to visualize.")
            return {}

        summary = self._build_solution_summary()
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
