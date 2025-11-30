from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd

# Allow execution both as a module (`python -m src.models.ortools_pipe.main`)
# and as a script (`python src/models/ortools_pipe/main.py`).
if __package__:
    from ..utils import get_adjacent_cells
    from .constants import ALPHA, BUDGET, SPECIES, TIME_LIMIT_MS
    from .cpsat import CPSATPipeModel
    from .solvers_lp import CBCPipeModel, SCIPPipeModel
else:  # pragma: no cover
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    SRC_ROOT = PROJECT_ROOT / "src"
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(SRC_ROOT))
    from src.models.ortools_pipe.constants import ALPHA, BUDGET, SPECIES, TIME_LIMIT_MS
    from src.models.ortools_pipe.cpsat import CPSATPipeModel
    from src.models.ortools_pipe.solvers_lp import CBCPipeModel, SCIPPipeModel
    from src.models.utils import get_adjacent_cells


def load_problem_data(
    data_path: Path,
) -> tuple[gpd.GeoDataFrame, dict, dict, dict, list, dict, dict]:
    df = gpd.read_parquet(data_path)
    print("Data loaded successfully")

    origin_cells_by_species = {}
    for species in SPECIES:
        column_name = f"has_{species}"
        species_cells = df[df[column_name]]["grid_id"].tolist()
        origin_cells_by_species[species] = species_cells

    all_cells = df["grid_id"].tolist()
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

    all_cells_set = set(all_cells)
    adjacency = {}
    for cell in all_cells:
        adjacency[cell] = get_adjacent_cells(cell, all_cells_set, df)

    return (
        df,
        origin_cells_by_species,
        cost_corridor_dict,
        cost_adaptation_dict,
        all_cells,
        adjacency,
        benefit_adaptation_dict,
    )


def run_pipeline(model_name: str, data_bundle: tuple) -> None:
    (
        df,
        origin_cells_by_species,
        cost_corridor_dict,
        cost_adaptation_dict,
        all_cells,
        adjacency,
        benefit_adaptation_dict,
    ) = data_bundle

    class_map = {
        "SCIP": SCIPPipeModel,
        "CBC": CBCPipeModel,
        "CPSAT": CPSATPipeModel,
    }

    model_cls = class_map[model_name]
    pipe_model = model_cls(
        corridor_cost=cost_corridor_dict,
        adaptation_cost=cost_adaptation_dict,
        adaptation_benefit=benefit_adaptation_dict,
        all_cells=all_cells,
        adjacency=adjacency,
        origin_cells_by_species=origin_cells_by_species,
        species_list=SPECIES.copy(),
        budget=BUDGET,
        alpha=ALPHA,
        time_limit_ms=TIME_LIMIT_MS,
    )

    if pipe_model.run_phase_1():
        if pipe_model.run_phase_2():
            pipe_model.run_phase_3()

    pipe_model.visualize_solution(df)


def main() -> None:
    root_path = Path(__file__).resolve().parents[3]
    data_path = root_path / "data" / "processed_dataset.parquet"

    data_bundle = load_problem_data(data_path)

    for model_name in ["CPSAT"]:
        print(f"\n{'#' * 80}\nRunning pipeline with {model_name} solver\n{'#' * 80}")

        initial_time = time.time()
        run_pipeline(model_name, data_bundle)
        total_elapsed_time = time.time() - initial_time

        print(
            f"\nTotal elapsed time for model {model_name}: {total_elapsed_time:.2f} seconds"
        )


if __name__ == "__main__":
    main()
