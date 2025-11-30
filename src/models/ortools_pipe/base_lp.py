from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import ortools.linear_solver.pywraplp as pywraplp

from ..utils import manhattan_distance
from ..visualization import (
    DEFAULT_SPECIES_COLORS,
    SolutionSummary,
    build_solution_summary,
    create_solution_map,
)
from .constants import (
    ALPHA,
    BUDGET,
    MAX_DISTANCE,
    NUM_THREADS,
    TIME_LIMIT_MS,
)
from .reporting import print_results


class ORToolsPipeModelBase:
    """
    Base pipeline implementation using the OR-Tools linear solver interface.
    Subclasses only implement `_create_solver` and solver-specific parameter setting.
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
        self.sol_x: dict = {}
        self.sol_y: dict = {}
        self.sol_u: dict = {}
        self.sol_rehab: dict = {}

        self.final_cost = 0.0
        self.final_model_status: int | None = None

    # REUSABLE MODEL BUILDING METHODS

    def _create_x_vars(self, model: pywraplp.Solver) -> dict:
        """Creates x variables (corridor built) for all cells."""
        x = {}
        for j in self.all_cells:
            x[j] = model.BoolVar(f"x_{j}")
        return x

    def _create_y_vars(self, model: pywraplp.Solver, species_list: list) -> dict:
        """Creates y variables (flow) for a list of species."""
        y = {}
        for species in species_list:
            origin_cells = self.origin_cells_by_species[species]
            for r in origin_cells:
                for j in self.all_cells:
                    if manhattan_distance(r, j) > MAX_DISTANCE:
                        continue
                    for k in self.adjacency[j]:
                        y[(species, r, j, k)] = model.BoolVar(
                            f"y_{species}_{r}_{j}_{k}"
                        )
        return y

    def _create_u_vars(self, model: pywraplp.Solver, species_list: list) -> dict:
        """Creates u variables (cell used) for a list of species."""
        u = {}
        for species in species_list:
            for j in self.all_cells:
                u[(species, j)] = model.BoolVar(f"u_{species}_{j}")
        return u

    def _create_rehab_vars(self, model: pywraplp.Solver, species_list: list) -> dict:
        """Creates rehab variables (rehabilitation) for a list of species."""
        rehab = {}
        for species in species_list:
            origin_cells = self.origin_cells_by_species[species]
            for j in self.all_cells:
                if j not in origin_cells:
                    rehab[(species, j)] = model.BoolVar(f"rehab_{species}_{j}")
        return rehab

    def _add_flow_constraints(
        self, model: pywraplp.Solver, y: dict, species_list: list
    ) -> None:
        """Adds outflow (1) and flow conservation (2) constraints."""
        print(f"Adding {len(species_list)} set(s) of flow constraints...")
        for species in species_list:
            origin_cells = self.origin_cells_by_species[species]
            # 1. Outflow
            for r in origin_cells:
                model.Add(
                    model.Sum(
                        y[(species, r, r, j)]
                        for j in self.adjacency[r]
                        if (species, r, r, j) in y
                    )
                    >= 1,
                    f"origin_outflow_{species}_{r}",
                )
            # 2. Flow conservation
            for j in self.all_cells:
                for r in origin_cells:
                    if j in origin_cells:
                        continue
                    inflow = model.Sum(
                        y[(species, r, i, j)]
                        for i in self.adjacency[j]
                        if (species, r, i, j) in y
                    )
                    outflow = model.Sum(
                        y[(species, r, j, k)]
                        for k in self.adjacency[j]
                        if k != r and (species, r, j, k) in y
                    )
                    model.Add(inflow - outflow == 0, f"flow_cons_{species}_{r}_{j}")

    def _add_no_reverse_flow_constraints(
        self, model: pywraplp.Solver, y: dict, species_list: list[str]
    ) -> None:
        """Prevent reverse flow between adjacent cells for the same origin."""
        print(f"Adding {len(species_list)} set(s) of no-reverse-flow constraints...")
        constraint_count = 0

        for species in species_list:
            origin_cells = self.origin_cells_by_species[species]
            for r in origin_cells:
                for j in self.all_cells:
                    for k in self.adjacency[j]:
                        if j < k:
                            if (species, r, j, k) in y and (species, r, k, j) in y:
                                model.Add(
                                    y[(species, r, j, k)] + y[(species, r, k, j)] <= 1
                                )
                                constraint_count += 1

        print(f"Total no-reverse-flow constraints added: {constraint_count}")

    def _add_activation_constraints(
        self, model: pywraplp.Solver, y: dict, u: dict, species_list: list
    ) -> None:
        """Adds cell activation constraints (3) (link y -> u)."""
        print(f"Adding {len(species_list)} set(s) of activation constraints...")
        for species in species_list:
            origin_cells = self.origin_cells_by_species[species]
            for j in self.all_cells:
                inflow_sum = model.Sum(
                    y[(species, r, i, j)]
                    for r in origin_cells
                    for i in self.adjacency[j]
                    if (species, r, i, j) in y
                )
                outflow_sum = model.Sum(
                    y[(species, r, j, k)]
                    for r in origin_cells
                    for k in self.adjacency[j]
                    if (species, r, j, k) in y
                )
                usage_sum = inflow_sum + outflow_sum

                # Big-M: Number of origins * (max inflow + max outflow edges)
                M_cell = len(origin_cells) * (len(self.adjacency[j]) * 2)
                M_cell = max(1, M_cell)  # Ensure M is at least 1

                model.Add(usage_sum <= M_cell * u[(species, j)])
                model.Add(u[(species, j)] <= usage_sum)

    def _add_fix_y_constraints(self, model: pywraplp.Solver, y: dict, sol_y: dict):
        """Adds constraints to fix Y variables to their previous solution."""
        print(f"Fixing {len(sol_y)} Y variables...")
        for k, v_sol in sol_y.items():
            if k in y:  # Ensure the key exists
                model.Add(y[k] == v_sol, f"fix_y_{k[0]}_{k[1]}_{k[2]}_{k[3]}")

    def _add_fix_u_constraints(self, model: pywraplp.Solver, u: dict, sol_u: dict):
        """Adds constraints to fix U variables to their previous solution."""
        print(f"Fixing {len(sol_u)} U variables...")
        for k, v_sol in sol_u.items():
            if k in u:  # Ensure the key exists
                model.Add(u[k] == v_sol, f"fix_u_{k[0]}_{k[1]}")

    # ABSTRACT METHODs (TO BE IMPLEMENTED BY SUBCLASSES)

    def _create_solver(self) -> pywraplp.Solver:
        """Return an OR-Tools solver instance."""
        raise NotImplementedError("Subclass must implement _create_solver()")

    def set_solver_parameters(
        self,
        solver: pywraplp.Solver,
        num_threads: int,
        time_limit_ms: int,
        gap: float = 0.2,
    ) -> None:
        """Configure solver parameters."""
        raise NotImplementedError("Subclass must implement set_solver_parameters()")

    # MAIN PIPELINE METHODS

    def run_phase_1(self) -> bool:
        print("\nStarting Phase 1: Atelerix Corridors")
        model = self._create_solver()
        self.set_solver_parameters(model, self.num_threads, self.time_limit_ms)

        species_f1 = ["atelerix_algirus"]

        # Phase 1 Variables
        x = self._create_x_vars(model)
        y = self._create_y_vars(model, species_f1)
        u = self._create_u_vars(model, species_f1)

        # Phase 1 Objective
        cost_expr = model.Sum(self.corridor_cost[j] * x[j] for j in self.all_cells)
        model.Minimize(cost_expr)

        # Phase 1 Constraints
        self._add_flow_constraints(model, y, species_f1)
        self._add_no_reverse_flow_constraints(model, y, species_f1)
        self._add_activation_constraints(model, y, u, species_f1)

        # 4. Link u and x (Simple version for Phase 1)
        for j in self.all_cells:
            model.Add(u[(species_f1[0], j)] <= x[j], f"link_u_x_{species_f1[0]}_{j}")

        # 5. Budget constraint
        model.Add(cost_expr <= self.budget, "budget_constraint")

        # Solve Phase 1
        start_time = time.time()
        status = model.Solve()
        print_results(model, status, time.time() - start_time, "Phase 1")

        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            print("Error: Phase 1 found no solution. Aborting.")
            return False

        # Save Phase 1 Solution
        self.sol_x = {j: v.SolutionValue() for j, v in x.items()}
        self.sol_y = {k: v.SolutionValue() for k, v in y.items()}
        self.sol_u = {k: v.SolutionValue() for k, v in u.items()}
        return True

    def run_phase_2(self) -> bool:
        print("\nStarting Phase 2: Remaining Corridors")
        model = self._create_solver()
        self.set_solver_parameters(model, self.num_threads, self.time_limit_ms)

        species_new = [s for s in self.full_species_list if s != "atelerix_algirus"]

        # Phase 2 Variables
        x = self._create_x_vars(model)
        y = self._create_y_vars(model, self.full_species_list)
        u = self._create_u_vars(model, self.full_species_list)

        # Phase 2 Objective
        cost_expr = model.Sum(self.corridor_cost[j] * x[j] for j in self.all_cells)
        model.Minimize(cost_expr)

        # Phase 2 Constraints
        # A. Fixing constraints (from Phase 1)
        self._add_fix_y_constraints(model, y, self.sol_y)
        self._add_fix_u_constraints(model, u, self.sol_u)

        # B. Constraints (1, 2, 3) for the *new* species
        self._add_flow_constraints(model, y, species_new)
        self._add_no_reverse_flow_constraints(model, y, species_new)
        self._add_activation_constraints(model, y, u, species_new)

        # C. New constraint (4) - Link all u and x
        for j in self.all_cells:
            lhs = model.Sum(u[(species, j)] for species in self.full_species_list)
            model.Add(lhs <= len(self.full_species_list) * x[j], f"link_u_x_all_{j}")

        # D. Constraint (5) - Budget (same)
        model.Add(cost_expr <= self.budget, "budget_constraint")

        martes_origin_cells = self.origin_cells_by_species.get("martes_martes", [])
        oryctolagus_origin_cells = self.origin_cells_by_species.get(
            "oryctolagus_cuniculus", []
        )
        eliomys_origin_cells = self.origin_cells_by_species.get("eliomys_quercinus", [])

        # E. Constraint (6) - Species compatibility
        for j in self.all_cells:
            if j in martes_origin_cells and (
                j in oryctolagus_origin_cells or j in eliomys_origin_cells
            ):
                continue

            term_martes = u.get(("martes_martes", j), 0)
            term_oryctolagus = u.get(("oryctolagus_cuniculus", j), 0)
            term_eliomys = u.get(("eliomys_quercinus", j), 0)
            model.Add(
                2 * term_martes + term_oryctolagus + term_eliomys <= 2,
                f"species_compatibility_{j}",
            )

        # Solve Phase 2
        start_time = time.time()
        status = model.Solve()
        print_results(model, status, time.time() - start_time, "Phase 2")

        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            print("Error: Phase 2 found no solution. Aborting.")
            return False

        # Save Phase 2 Solution
        self.sol_x = {j: v.SolutionValue() for j, v in x.items()}
        self.sol_y = {k: v.SolutionValue() for k, v in y.items()}
        self.sol_u = {k: v.SolutionValue() for k, v in u.items()}
        return True

    def run_phase_3(self) -> bool:
        print("\nStarting Phase 3: Habitat Rehabilitation")

        # PHASE 3 MODEL LOGIC
        model = self._create_solver()  # Start fresh
        self.set_solver_parameters(model, self.num_threads, self.time_limit_ms)

        # Variables:
        x = self._create_x_vars(model)
        y = self._create_y_vars(model, self.full_species_list)
        u = self._create_u_vars(model, self.full_species_list)
        rehab = self._create_rehab_vars(model, self.full_species_list)

        # Objective Phase 3
        cost_corridor = model.Sum(self.corridor_cost[j] * x[j] for j in self.all_cells)
        cost_adaptation = model.Sum(
            self.adaptation_cost[k] * v for k, v in rehab.items()
        )
        benefit_adaptation = model.Sum(
            self.adaptation_benefit[k] * v for k, v in rehab.items()
        )

        self.total_cost_expr = cost_corridor + cost_adaptation

        model.Minimize(
            (self.alpha * self.total_cost_expr)
            - ((1 - self.alpha) * benefit_adaptation)
        )

        # Phase 3 Constraints
        # A. Fixing constraints (from Phase 2)
        self._add_fix_y_constraints(model, y, self.sol_y)
        self._add_fix_u_constraints(model, u, self.sol_u)

        # B. Constraints from Phase 2 that persist
        # 4. Link all u and x
        for j in self.all_cells:
            lhs = model.Sum(u[(species, j)] for species in self.full_species_list)
            model.Add(lhs <= len(self.full_species_list) * x[j], f"link_u_x_all_{j}")

        martes_origin_cells = self.origin_cells_by_species.get("martes_martes", [])
        oryctolagus_origin_cells = self.origin_cells_by_species.get(
            "oryctolagus_cuniculus", []
        )
        eliomys_origin_cells = self.origin_cells_by_species.get("eliomys_quercinus", [])

        # E. Constraint (6) - Species compatibility
        for j in self.all_cells:
            if j in martes_origin_cells and (
                j in oryctolagus_origin_cells or j in eliomys_origin_cells
            ):
                continue

            term_martes = u.get(("martes_martes", j), 0)
            term_oryctolagus = u.get(("oryctolagus_cuniculus", j), 0)
            term_eliomys = u.get(("eliomys_quercinus", j), 0)
            model.Add(
                2 * term_martes + term_oryctolagus + term_eliomys <= 2,
                f"species_compatibility_u_{j}",
            )

        # C. New Phase 3 constraints
        # 5. New Budget Constraint
        model.Add(self.total_cost_expr <= self.budget, "budget_constraint_final")

        # 7. Rehab adjacency
        for (species, j), var in rehab.items():
            adj_usage = model.Sum(
                self.sol_u[(species, k)]
                for k in self.adjacency[j]
                if (species, k) in self.sol_u
            )
            model.Add(var <= adj_usage, f"rehab_adjacent_{species}_{j}")

        # 8. Rehab compatibility
        for j in self.all_cells:
            term_martes = rehab.get(("martes_martes", j), 0)
            term_oryctolagus = rehab.get(("oryctolagus_cuniculus", j), 0)
            term_eliomys = rehab.get(("eliomys_quercinus", j), 0)
            model.Add(
                2 * term_martes + term_oryctolagus + term_eliomys <= 2,
                f"rehab_compatibility_{j}",
            )

        # Solve Phase 3
        start_time = time.time()
        status = model.Solve()
        elapsed_time = time.time() - start_time
        self.final_model_status = status

        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            print("Error: Phase 3 found no solution.")
            return False

        # Save Final Solution
        self.sol_x = {j: v.SolutionValue() for j, v in x.items()}
        self.sol_y = {k: v.SolutionValue() for k, v in y.items()}
        self.sol_u = {k: v.SolutionValue() for k, v in u.items()}
        self.sol_rehab = {k: v.SolutionValue() for k, v in rehab.items()}

        # Calculate final cost
        self.final_cost = sum(
            self.corridor_cost[j] * self.sol_x[j] for j in self.all_cells
        ) + sum(self.adaptation_cost[k] * v for k, v in self.sol_rehab.items())

        print_results(model, status, elapsed_time, "Phase 3", self.final_cost)

        return True

    def _build_solution_summary(self) -> SolutionSummary:
        """Collect normalized solution data for visualization."""
        built_corridors = {j for j, value in self.sol_x.items() if value > 0.5}
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
        map_filename: str = "ortools_pipe_lp.html",
    ) -> None:
        # Default to maps directory in project root
        if save_dir is None:
            save_dir = Path(__file__).parent.parent.parent.parent / "maps"
        # Create save directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.final_model_status not in [
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
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
            title="Legend - OR-Tools Model",
        )

        save_path = save_dir / map_filename
        folium_map.save(save_path)
        print(f"\n✓ Map saved to: {save_path}")

    def visualize_solution_per_species(
        self,
        df: gpd.GeoDataFrame,
        save_dir: Path | None = None,
        map_filename_prefix: str = "ortools_pipe_lp",
    ) -> dict[str, Path]:
        # Default to maps directory in project root
        if save_dir is None:
            save_dir = Path(__file__).parent.parent.parent.parent / "maps"
        """
        Create one map per species highlighting its corridors and adaptations.
        Returns a mapping of species to saved file paths.
        """
        if self.final_model_status not in [
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
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
