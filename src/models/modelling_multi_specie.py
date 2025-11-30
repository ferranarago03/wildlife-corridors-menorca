import json
import pathlib
import sys
import time

import geopandas as gpd
import gurobipy as gp
from gurobipy import GRB

# Add src to path to enable imports when running as script or importing
src_path = pathlib.Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from models.gurobi.constants import (
    ALPHA,
    BUDGET,
    FOCUS,
    GAP,
    HEURISTICS,
    MIN_COVERAGE_FRACTION,
    PENALTY_UNCOVERED_ORIGIN,
    SPECIES,
)
from models.utils import get_adjacent_cells
from models.visualization import (
    build_solution_summary,
    create_solution_map,
    save_solution_summary,
)

# ============================================================================
# SCRIPT CONSTANTS
# ============================================================================

TIME_LIMIT_SECONDS = 60 * 30

# Set this to a pathlib.Path or string to control where variable values are stored.
# Example: VARIABLE_VALUES_OUTPUT_PATH = pathlib.Path("/tmp/solution_vars.json")
VARIABLE_VALUES_OUTPUT_PATH: pathlib.Path | str | None = pathlib.Path(
    "./data/experiments/variable_values/modelling_multi_species_variable_values.json"
)

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

# ============================================================================
# DATA LOADING
# ============================================================================

root_path = pathlib.Path(__file__).parent.parent.parent
data_path = root_path / "data" / "processed_dataset.parquet"
df = gpd.read_parquet(data_path)

# ============================================================================
# SPECIES AND ORIGIN CELL SELECTION
# ============================================================================


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
print("\nOrigin cells by species:")
for species in SPECIES:
    print(f"  {species}:")
    print(f"    - Origins: {len(origin_cells_by_species[species])}")
    if len(origin_cells_by_species[species]) > 0:
        print(f"    - First 5 cells: {origin_cells_by_species[species][:5]}")

# ============================================================================
# ADJACENCY COMPUTATION
# ============================================================================

all_cells_set = set(all_cells)
adjacency = {}

for cell in all_cells:
    adjacency[cell] = get_adjacent_cells(cell, all_cells_set, df)

print("\nExample adjacencies:")
for i, cell in enumerate(all_cells[:3]):
    print(f"{cell}: {adjacency[cell]}")
    geom_cell = df[df["grid_id"] == cell]["geometry"].values[0]
    for j, neighbor in enumerate(adjacency[cell]):
        geom_neighbor = df[df["grid_id"] == neighbor]["geometry"].values[0]
        print(
            "Touches:",
            neighbor,
            geom_cell.touches(geom_neighbor),
        )

# ============================================================================
# CONSTANTS DICTIONARIES
# ============================================================================

cost_corridor_dict = dict(zip(df["grid_id"], df["cost_corridor"]))

print("\nExample corridor costs:")
for cell in all_cells[:5]:
    print(f"{cell}: {cost_corridor_dict[cell]:.2f}")

cost_adaptation_dict = {}
for j in all_cells:
    for species in SPECIES:
        cost_adaptation_dict[(species, j)] = df[df["grid_id"] == j][
            f"cost_adaptation_{species.split('_')[0]}"
        ].values[0]

print("\nExample adaptation costs:")
for species in SPECIES:
    for cell in all_cells[:5]:
        print(f"{species}, {cell}: {cost_adaptation_dict[(species, cell)]:.2f}")

benefit_adaptation_dict = {}
for j in all_cells:
    for species in SPECIES:
        benefit_adaptation_dict[(species, j)] = df[df["grid_id"] == j][
            f"{species.split('_')[1]}_benefit"
        ].values[0]

print("\nExample adaptation benefits:")
for species in SPECIES:
    for cell in all_cells[:5]:
        print(f"{species}, {cell}: {benefit_adaptation_dict[(species, cell)]:.2f}")

# ============================================================================
# BUDGET SHARES
# ============================================================================


def _prepare_budget_shares(
    species_list, total_budget, corridor_shares, adaptation_shares
):
    corridor_share_total = sum(corridor_shares.get(sp, 0.0) for sp in species_list)
    if corridor_share_total > 1.0 + 1e-9:
        raise ValueError(
            f"Corridor shares sum to {corridor_share_total:.3f} > 1.0; reduce CORRIDOR_SHARE_BY_SPECIES."
        )

    if adaptation_shares is None:
        remaining = max(0.0, 1.0 - corridor_share_total)
        adaptation_shares = {sp: remaining / len(species_list) for sp in species_list}
    adaptation_share_total = sum(adaptation_shares.get(sp, 0.0) for sp in species_list)

    if corridor_share_total + adaptation_share_total > 1.0 + 1e-9:
        raise ValueError(
            f"Corridor+adaptation shares sum to {corridor_share_total + adaptation_share_total:.3f} > 1.0; reduce percentages."
        )

    corridor_budget_by_species = {
        sp: total_budget * corridor_shares.get(sp, 0.0) for sp in species_list
    }
    adaptation_budget_by_species = {
        sp: total_budget * adaptation_shares.get(sp, 0.0) for sp in species_list
    }
    return (
        corridor_budget_by_species,
        adaptation_budget_by_species,
        corridor_share_total,
        adaptation_share_total,
    )


(
    corridor_budget_by_species,
    adaptation_budget_by_species,
    corridor_share_total,
    adaptation_share_total,
) = _prepare_budget_shares(
    SPECIES, BUDGET, CORRIDOR_SHARE_BY_SPECIES, ADAPTATION_SHARE_BY_SPECIES
)

print("=" * 60)
print(
    "\nBudgets configured:"
    f"\n  Total: {BUDGET:.2f}"
    f"\n  Corridors (sum shares): {corridor_share_total:.2f}"
    f"\n  Adaptation (sum shares): {adaptation_share_total:.2f}"
)
for sp in SPECIES:
    print(
        f"    {sp}: corridors {corridor_budget_by_species.get(sp, 0.0):.2f} | "
        f"adaptation {adaptation_budget_by_species.get(sp, 0.0):.2f}"
    )

print("=" * 60)
# ============================================================================
# MODEL CREATION
# ============================================================================

model = gp.Model("WildlifeCorridors")

print(f"\n   Gurobi version: {gp.gurobi.version()}")

# ============================================================================
# DECISION VARIABLES
# ============================================================================

x = {}
for j in all_cells:
    x[j] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}")

print(f"Number of corridor variables (x): {len(x)}")


y = {}
for species in SPECIES:
    origin_cells = origin_cells_by_species[species]
    for r in origin_cells:
        for j in all_cells:
            for k in adjacency[j]:
                y[(species, r, j, k)] = model.addVar(
                    vtype=GRB.BINARY, name=f"y_{species}_{r}_{j}_{k}"
                )

print(f"Number of flow variables (y): {len(y)}")
print("  Variables per species:")
for species in SPECIES:
    species_vars = sum(1 for key in y.keys() if key[0] == species)
    print(f"    {species}: {species_vars}")


covered = {}
for species in SPECIES:
    for origin in origin_cells_by_species[species]:
        covered[(species, origin)] = model.addVar(
            vtype=GRB.BINARY, name=f"covered_{species}_{origin}"
        )

u = {}
rehab = {}
for species in SPECIES:
    origin_cells = origin_cells_by_species[species]
    for j in all_cells:
        u[(species, j)] = model.addVar(vtype=GRB.BINARY, name=f"u_{species}_{j}")

        if j not in origin_cells:
            rehab[(species, j)] = model.addVar(
                vtype=GRB.BINARY, name=f"rehab_{species}_{j}"
            )

print(f"Number of auxiliary variables (u): {len(u)}")

print(f"Number of rehabilitation variables (rehab): {len(rehab)}")


model.update()

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

cost = gp.LinExpr()
benefit = gp.LinExpr()

for j in x.keys():
    cost += cost_corridor_dict[j] * x[j]
    for species in SPECIES:
        if (species, j) in rehab:
            cost += cost_adaptation_dict[(species, j)] * rehab[(species, j)]

for j in x.keys():
    for species in SPECIES:
        if (species, j) in rehab:
            benefit += benefit_adaptation_dict[(species, j)] * rehab[(species, j)]

if PENALTY_UNCOVERED_ORIGIN is not None:
    for var in covered.values():
        cost += PENALTY_UNCOVERED_ORIGIN * (1 - var)

model.setObjective((ALPHA * cost) - ((1 - ALPHA) * benefit), GRB.MINIMIZE)

print("Objective function set: Minimize total corridor construction cost")

# ============================================================================
# CONSTRAINTS
# ============================================================================

print("\nAdding constraint (2): Optional outgoing corridor per origin...")

constraint_count = 0
for species in SPECIES:
    origin_cells = origin_cells_by_species[species]
    for r in origin_cells:
        outflow = gp.LinExpr()
        deg_r = max(1, len(adjacency[r]))

        for j in adjacency[r]:
            outflow += y[(species, r, r, j)]

        model.addConstr(
            outflow >= covered[(species, r)],
            name=f"origin_outflow_min_{species}_{r}",
        )
        model.addConstr(
            outflow <= deg_r * covered[(species, r)],
            name=f"origin_outflow_max_{species}_{r}",
        )
        constraint_count += 2

        for j in all_cells:
            for k in adjacency[j]:
                if (species, r, j, k) in y:
                    model.addConstr(
                        y[(species, r, j, k)] <= covered[(species, r)],
                        name=f"origin_flow_bound_{species}_{r}_{j}_{k}",
                    )
                    constraint_count += 1

print(f"Added {constraint_count} origin coverage constraints")

if MIN_COVERAGE_FRACTION is not None and covered:
    print("Adding constraint (15): Minimum coverage fraction...")
    total_origins = len(covered)
    required = MIN_COVERAGE_FRACTION * total_origins
    model.addConstr(
        gp.quicksum(covered.values()) >= required,
        name="min_coverage_fraction",
    )
    print(
        f"Added minimum coverage constraint requiring {MIN_COVERAGE_FRACTION:.2%} of origins"
    )

# ============================================================================

print("\nAdding constraint (4): Flow conservation at intermediate nodes...")

constraint_count = 0

for species in SPECIES:
    origin_cells = origin_cells_by_species[species]

    for r in origin_cells:
        for j in all_cells:
            if j in origin_cells:
                continue

            flow_balance = gp.LinExpr()

            for i in adjacency[j]:
                flow_balance += y[(species, r, i, j)]

            for k in adjacency[j]:
                if k != r:
                    flow_balance -= y[(species, r, j, k)]

            model.addConstr(
                flow_balance == 0, name=f"flow_conservation_{species}_{r}_{j}"
            )
            constraint_count += 1

print(f"Added {constraint_count} flow conservation constraints")

# ============================================================================

print(f"Adding {len(SPECIES)} set(s) of no-reverse-flow constraints...")

constraint_count = 0

for species in SPECIES:
    origin_cells = origin_cells_by_species[species]

    for r in origin_cells:
        for j in all_cells:
            for k in adjacency[j]:
                if j < k:
                    if (species, r, j, k) in y and (species, r, k, j) in y:
                        model.addConstr(
                            y[(species, r, j, k)] + y[(species, r, k, j)] <= 1,
                            name=f"no_reverse_flow_{species}_{r}_{j}_{k}",
                        )
                        constraint_count += 1

print(f"Total no-reverse-flow constraints added: {constraint_count}")

# ============================================================================

print("\nAdding constraint (6): Flow only on built corridors...")

constraint_count = 0

for species in SPECIES:
    for j in all_cells:
        flow_sum = gp.LinExpr()
        m_cell_count = 0
        for r in origin_cells_by_species[species]:
            for k in adjacency[j]:
                if (species, r, j, k) in y:
                    flow_sum += y[(species, r, j, k)]
                    flow_sum += y[(species, r, k, j)]
                    m_cell_count += 2

        M_cell = max(1, m_cell_count)
        model.addConstr(
            flow_sum <= M_cell * u[(species, j)], name=f"flow_on_built_{species}_{j}"
        )
        model.addConstr(
            u[(species, j)] <= flow_sum, name=f"built_if_flow_{species}_{j}"
        )
        constraint_count += 2

print(
    f"Added {constraint_count} aggregated flow capacity constraints (per species, both directions)"
)

# ============================================================================

print("\nAdding constraint (7): Linking u and x variables...")

num_constraints = 0
for j in all_cells:
    species_count = 0
    lhs = gp.LinExpr()
    for species in SPECIES:
        lhs += u[(species, j)]
        species_count += 1
    M_species = species_count
    model.addConstr(lhs <= M_species * x[j], name=f"u_x_link_{j}")
    num_constraints += 1

print(f"Added {num_constraints} constraints linking u and x variables")


# ===========================================================================

print(
    "\nAdding constraint (8): Incompatibility Martes martes - Oryctolagus cuniculus and Eliomys quercinus"
)

constraint_count = 0

martes_origin_cells = origin_cells_by_species.get("martes_martes", [])
oryctolagus_origin_cells = origin_cells_by_species.get("oryctolagus_cuniculus", [])
eliomys_origin_cells = origin_cells_by_species.get("eliomys_quercinus", [])

for j in all_cells:
    if j in martes_origin_cells and (
        j in oryctolagus_origin_cells or j in eliomys_origin_cells
    ):
        continue

    model.addConstr(
        2 * u.get(("martes_martes", j), 0)
        + u.get(("oryctolagus_cuniculus", j), 0)
        + u.get(("eliomys_quercinus", j), 0)
        <= 2,
        name=f"incompatibility_martes_oryctolagus_{j}",
    )
    constraint_count += 1

print(
    f"Added {constraint_count} incompatibility constraints Martes martes - Oryctolagus cuniculus"
)

# ===========================================================================

print("\nAdding constraint (9): Only adapt if adyacent corridor or origin is built")

constraint_count = 0
for species in SPECIES:
    origins = origin_cells_by_species[species]
    for j in all_cells:
        if (species, j) not in rehab:
            continue
        adj_corridors = gp.LinExpr()

        touches_origin = 0
        for k in adjacency[j]:
            adj_corridors += u[(species, k)]
            if k in origins:
                touches_origin = 1

        model.addConstr(
            rehab[(species, j)] <= adj_corridors + touches_origin,
            name=f"rehab_adjacent_{species}_{j}",
        )
        constraint_count += 1

print(f"Added {constraint_count} adaptation adjacency constraints")


# ============================================================================

print("\nAdding constraint (10): Adaptation compatibility")

constraint_count = 0
for j in all_cells:
    model.addConstr(
        2 * rehab.get(("martes_martes", j), 0)
        + rehab.get(("oryctolagus_cuniculus", j), 0)
        + rehab.get(("eliomys_quercinus", j), 0)
        <= 2,
        name=f"rehab_compatibility_{j}",
    )
    constraint_count += 1

print(f"Added {constraint_count} one-species-per-adapted-cell constraints")

# ============================================================================

print(
    "\nAdding constraint (11): Corridor for martes or Adaptation for oryctolagus and eliomys"
)

constraint_count = 0
for j in all_cells:
    model.addConstr(
        2 * u.get(("martes_martes", j), 0)
        + rehab.get(("oryctolagus_cuniculus", j), 0)
        + rehab.get(("eliomys_quercinus", j), 0)
        <= 2,
        name=f"corridor_adaptation_compatibility_{j}",
    )
    constraint_count += 1

print(f"Added {constraint_count} corridor-adaptation compatibility constraints")

# ============================================================================

print(
    "\nAdding constraint (12): Adaptation for martes or Corridor for oryctolagus and eliomys"
)

constraint_count = 0
for j in all_cells:
    model.addConstr(
        2 * rehab.get(("martes_martes", j), 0)
        + u.get(("oryctolagus_cuniculus", j), 0)
        + u.get(("eliomys_quercinus", j), 0)
        <= 2,
        name=f"adaptation_corridor_compatibility_{j}",
    )
    constraint_count += 1

print(f"Added {constraint_count} adaptation-corridor compatibility constraints")

# ============================================================================
print("\nAdding constraint (13): Budget constraint")


print("\nAdding constraint (13a): Per-species corridor budgets")
for species in SPECIES:
    cap = corridor_budget_by_species.get(species, 0.0)
    corridor_cost_species = gp.LinExpr()
    for j in all_cells:
        corridor_cost_species += cost_corridor_dict[j] * u[(species, j)]
    model.addConstr(corridor_cost_species <= cap, name=f"budget_corridor_{species}")

print("Adding constraint (13b): Per-species adaptation budgets")
for species in SPECIES:
    cap = adaptation_budget_by_species.get(species, 0.0)
    adaptation_cost_species = gp.LinExpr()
    for j in all_cells:
        if (species, j) in rehab:
            adaptation_cost_species += (
                cost_adaptation_dict[(species, j)] * rehab[(species, j)]
            )
    model.addConstr(adaptation_cost_species <= cap, name=f"budget_adaptation_{species}")


# ============================================================================
# SOLVE THE MODEL
# ============================================================================

print("\n" + "=" * 60)
print("SOLVING THE MODEL")
print("=" * 60)


# --- PERFORMANCE CONFIGURATION ---

model.setParam("TimeLimit", TIME_LIMIT_SECONDS)

model.setParam("MIPGap", GAP)

model.setParam("MIPFocus", FOCUS)

model.setParam("Heuristics", HEURISTICS)

# --- END OF PERFORMANCE CONFIGURATION ---

print(
    f"Time limit: {TIME_LIMIT_SECONDS} seconds ({TIME_LIMIT_SECONDS / 60:.1f} minutes)"
)
print("Threads: Using all available threads (configuration = 0)")
print("MIPGap: Stop at 5% optimality")
print("MIPFocus: 1 (Find feasible solutions)")


start_time = time.time()

model.optimize()

elapsed_time = time.time() - start_time

# Calculate the total cost with the obtained solution
total_cost = 0.0
if model.SolCount > 0:
    for j in x.keys():
        if x[j].X > 0.5:
            total_cost += cost_corridor_dict[j]
        for species in SPECIES:
            if (species, j) in rehab and rehab[(species, j)].X > 0.5:
                total_cost += cost_adaptation_dict[(species, j)]

print(f"\n{'=' * 60}")
print("SOLUTION RESULTS")
print(f"{'=' * 60}")
print(f"Solution status: {model.Status}")

if model.Status == GRB.OPTIMAL:
    print("✓ OPTIMAL solution found!")
elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
    print("✓ FEASIBLE solution found (time limit reached, not proven optimal)")
elif model.Status in [GRB.SUBOPTIMAL] or model.SolCount > 0:
    print("✓ FEASIBLE solution found (not proven optimal)")
else:
    print("✗ No solution found")

if model.SolCount > 0:
    print(f"\nObjective value: {model.ObjVal:.2f}")
    print(f"Total cost: {total_cost:.2f}")
    print(f"Number of variables: {model.NumVars}")
    print(f"Number of constraints: {model.NumConstrs}")
    print(f"Solver runtime: {model.Runtime:.2f} seconds")
    print(f"Actual elapsed time: {elapsed_time:.2f} seconds")

# ============================================================================
# EXTRACT AND DISPLAY SOLUTION
# ============================================================================

if model.SolCount > 0:
    built_corridors = [j for j, var in x.items() if var.X > 0.5]

    print(f"\n{'=' * 60}")
    print("CORRIDORS USED PER SPECIES")
    print(f"{'=' * 60}")

    species_using_corridor = {cell_id: [] for cell_id in built_corridors}

    for species in SPECIES:
        origin_cells = origin_cells_by_species[species]
        corridors_used = set()

        for r in origin_cells:
            for j in all_cells:
                for k in adjacency[j]:
                    if (species, r, j, k) in y and y[(species, r, j, k)].X > 0.5:
                        corridors_used.add(j)
                        if (
                            j in built_corridors
                            and species not in species_using_corridor[j]
                        ):
                            species_using_corridor[j].append(species)

        print(f"\n{species}:")
        print(f"  Origins: {len(origin_cells)}")
        print(f"  Corridors used: {len(corridors_used)}")

    species_rehabilitated = {}
    for (species, j), var in rehab.items():
        if var.X > 0.5:
            if j not in species_rehabilitated:
                species_rehabilitated[j] = []
            species_rehabilitated[j].append(species)

    print(f"\nTotal built corridors: {len(built_corridors)}")
    print(f"Total rehabilitated cells: {len(species_rehabilitated)}")

    species_colors = {
        "oryctolagus_cuniculus": "green",
        "atelerix_algirus": "purple",
        "eliomys_quercinus": "blue",
        "martes_martes": "red",
    }

    summary = build_solution_summary(
        SPECIES,
        origin_cells_by_species,
        built_corridors,
        y,
        rehab,
        species_colors,
    )
    summary_path = (
        root_path
        / "data"
        / "experiments"
        / "summaries"
        / f"modelling_multi_species_alpha_{ALPHA}_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_solution_summary(summary, summary_path)

    maps_dir = pathlib.Path(__file__).parent.parent.parent / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Map for all species
    map_all = create_solution_map(
        df,
        summary,
        species=None,
        title="Multi-species corridors - All species",
    )
    map_all_path = maps_dir / f"modelling_multi_species_results_all_alpha_{ALPHA}.html"
    map_all.save(str(map_all_path))
    print(f"\n✓ Map for all species saved to: {map_all_path}")

    # Map for each species
    for sp in SPECIES:
        map_sp = create_solution_map(
            df,
            summary,
            species=sp,
            title=f"Multi-species corridors - {sp}",
        )
        map_sp_path = (
            maps_dir / f"modelling_multi_species_results_{sp}_alpha_{ALPHA}.html"
        )
        map_sp.save(str(map_sp_path))
        print(f"✓ Map for {sp} saved to: {map_sp_path}")

    # Save the value of each variable in a json
    if VARIABLE_VALUES_OUTPUT_PATH is not None:
        output_path = pathlib.Path(VARIABLE_VALUES_OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _serialize_vars(var_dict):
            serialized = {}
            for key, var in var_dict.items():
                if isinstance(key, tuple):
                    key_name = "|".join(str(part) for part in key)
                else:
                    key_name = str(key)
                serialized[key_name] = float(var.X)
            return serialized

        variable_values = {
            "x": _serialize_vars(x),
            "y": _serialize_vars(y),
            "u": _serialize_vars(u),
            "rehab": _serialize_vars(rehab),
            "covered": _serialize_vars(covered),
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(variable_values, f, indent=2)

        print(f"✓ Variable values saved to: {output_path}")
    else:
        print(
            "Variable values output path not set; define VARIABLE_VALUES_OUTPUT_PATH to store them."
        )


print("\n" + "=" * 60)
print("SCRIPT COMPLETED")
print("=" * 60)
