from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import networkx as nx
import numpy as np

from models.utils import manhattan_distance
from models.visualization import DEFAULT_SPECIES_COLORS, SolutionSummary


class MenorcaEcologicalCorridor:
    """
    Handles graph construction and enumeration of minimum-cost paths
    between origins for each species using Dijkstra's algorithm.
    """

    def __init__(
        self,
        costs: dict[str, float],
        adjacency: dict[str, list[str]],
        gdf: gpd.GeoDataFrame,
        origins: dict[str, list[str]],
    ):
        self.costs = costs
        self.adjacency = adjacency
        self.gdf = gdf
        self.origins = origins
        self.graph: nx.DiGraph | None = None
        self.optimal_paths: dict[tuple[str, str, str], list[str]] = {}

    def build_graph(self, excluded_cells: set[str] | None = None) -> nx.DiGraph:
        G = nx.DiGraph()
        excluded_cells = excluded_cells or set()

        for cell_id, cost in self.costs.items():
            if cost < np.inf and cell_id not in excluded_cells:
                G.add_node(cell_id, cost=cost)

        for cell_id, neighbors in self.adjacency.items():
            if not G.has_node(cell_id):
                continue
            for neighbor_id in neighbors:
                if G.has_node(neighbor_id):
                    weight = self.costs.get(neighbor_id, np.inf)
                    if np.isfinite(weight):
                        G.add_edge(cell_id, neighbor_id, weight=weight)

        self.graph = G
        print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def compute_paths_by_species(
        self, species: str, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str, str], list[str]]:
        if self.graph is None:
            raise ValueError("You must first build the graph with build_graph().")

        if species not in self.origins:
            print(f"Warning: species '{species}' not found in origins")
            return {}

        results: dict[tuple[str, str, str], list[str]] = {}
        paths_found = 0

        for start, end in pairs:
            if not self.graph.has_node(start) or not self.graph.has_node(end):
                continue
            try:
                path = nx.dijkstra_path(self.graph, start, end, weight="weight")
            except nx.NetworkXNoPath:
                continue

            key = (species, start, end)
            results[key] = path
            self.optimal_paths[key] = path
            paths_found += 1

        if paths_found == 0:
            raise ValueError(
                f"No paths found for {species}. Try with a larger Manhattan distance."
            )

        print(
            f"Species {species}: {paths_found} paths from {len(pairs)} pairs evaluated."
        )
        return results

    def get_path_info(self, species: str, start: str, end: str) -> dict[str, Any]:
        key = (species, start, end)
        if key not in self.optimal_paths:
            return {}

        path = self.optimal_paths[key]
        total_cost = sum(self.costs[cell] for cell in path[1:])
        return {
            "species": species,
            "start": start,
            "end": end,
            "path": path,
            "cost": total_cost,
            "length": len(path) - 1,
            "num_cells": len(path),
        }


@dataclass
class PathCandidate:
    path_id: str
    species: str
    origin: str
    destination: str
    cells: list[str]
    cost: float
    length: int


def generate_pairs(origins: Sequence[str], max_manhattan: int) -> list[tuple[str, str]]:
    origins_sorted = sorted(origins)
    pairs: list[tuple[str, str]] = []
    for i in range(len(origins_sorted)):
        for j in range(i + 1, len(origins_sorted)):
            ini = origins_sorted[i]
            fin = origins_sorted[j]
            if manhattan_distance(ini, fin) <= max_manhattan:
                pairs.append((ini, fin))
    return pairs


def enumerate_candidates(
    graph: MenorcaEcologicalCorridor,
    species_list: Iterable[str],
    max_manhattan_distance: int,
) -> tuple[list[PathCandidate], dict[str, set[str]], dict[tuple[str, str], set[str]]]:
    candidates: list[PathCandidate] = []
    cell_to_paths: dict[str, set[str]] = defaultdict(set)
    origin_to_paths: dict[tuple[str, str], set[str]] = defaultdict(set)

    for species in species_list:
        origin_cells = graph.origins.get(species, [])
        pairs = generate_pairs(origin_cells, max_manhattan_distance)
        print(f"\nEnumerating paths for {species}: {len(pairs)} pairs.")
        if not pairs:
            continue
        try:
            paths = graph.compute_paths_by_species(
                species=species,
                pairs=pairs,
            )
        except ValueError as exc:
            print(f"  -> No paths generated: {exc}")
            continue

        for (sp, start, end), path in paths.items():
            info = graph.get_path_info(sp, start, end)
            if not info:
                continue
            path_id = f"{sp}__{start}__{end}"
            candidate = PathCandidate(
                path_id=path_id,
                species=sp,
                origin=start,
                destination=end,
                cells=list(path),
                cost=float(info["cost"]),
                length=int(info["length"]),
            )
            candidates.append(candidate)
            for cell in candidate.cells:
                cell_to_paths[cell].add(path_id)
            origin_to_paths[(sp, start)].add(path_id)
            origin_to_paths[(sp, end)].add(path_id)

    print(
        f"Total candidate paths: {len(candidates)} covering {len(cell_to_paths)} cells."
    )
    return candidates, cell_to_paths, origin_to_paths


def build_cost_dicts(
    df: gpd.GeoDataFrame, all_cells: Sequence[str], species_list: Sequence[str]
) -> tuple[
    dict[str, float], dict[tuple[str, str], float], dict[tuple[str, str], float]
]:
    cost_corridor_dict = dict(zip(df["grid_id"], df["cost_corridor"]))
    cost_adaptation_dict: dict[tuple[str, str], float] = {}
    benefit_adaptation_dict: dict[tuple[str, str], float] = {}

    for cell in all_cells:
        row = df[df["grid_id"] == cell].iloc[0]
        for species in species_list:
            prefix = species.split("_")[0]
            suffix = species.split("_")[1]
            cost_adaptation_dict[(species, cell)] = row[f"cost_adaptation_{prefix}"]
            benefit_adaptation_dict[(species, cell)] = row[f"{suffix}_benefit"]

    return cost_corridor_dict, cost_adaptation_dict, benefit_adaptation_dict


def report_selection(
    title: str,
    selected_by_species: Mapping[str, list[PathCandidate]],
    species_list: Sequence[str],
    max_examples: int = 5,
) -> None:
    print(f"\n{title}")
    total = 0
    for species in species_list:
        paths = selected_by_species.get(species, [])
        total += len(paths)
        print(f"  {species}: {len(paths)} paths")
        for candidate in paths[:max_examples]:
            print(
                f"    - {candidate.origin} -> {candidate.destination} | "
                f"cost {candidate.cost:.2f} | length {candidate.length}"
            )
    print(f"  Total: {total} paths selected")


def collect_used_cells(
    selection_a: Mapping[str, list[PathCandidate]],
    selection_b: Mapping[str, list[PathCandidate]],
    species_list: Sequence[str],
) -> dict[str, set[str]]:
    used: dict[str, set[str]] = {sp: set() for sp in species_list}
    for selections in (selection_a, selection_b):
        for sp, candidates in selections.items():
            used.setdefault(sp, set()).update(
                cell for cand in candidates for cell in cand.cells
            )
    return used


def build_summary_from_paths(
    selected_phase1: Mapping[str, list[PathCandidate]],
    selected_martes: Mapping[str, list[PathCandidate]],
    origin_cells_by_species: Mapping[str, list[str]],
    species_list: Sequence[str],
    rehab_selected: Mapping[str, set[str]] | None = None,
    species_colors: Mapping[str, str] | None = None,
) -> SolutionSummary:
    colors = species_colors or DEFAULT_SPECIES_COLORS
    corridors_by_species: dict[str, set[str]] = {sp: set() for sp in species_list}
    built_corridors: set[str] = set()

    for selected in (selected_phase1, selected_martes):
        for sp, candidates in selected.items():
            for cand in candidates:
                corridors_by_species.setdefault(sp, set()).update(cand.cells)
                built_corridors.update(cand.cells)

    if rehab_selected is None:
        rehabilitated_by_species = {sp: set() for sp in species_list}
    else:
        rehabilitated_by_species = {
            sp: set(rehab_selected.get(sp, set())) for sp in species_list
        }

    return SolutionSummary(
        species_list=list(species_list),
        built_corridors=built_corridors,
        corridors_by_species=corridors_by_species,
        rehabilitated_by_species=rehabilitated_by_species,
        origin_cells_by_species={
            sp: list(origin_cells_by_species.get(sp, [])) for sp in species_list
        },
        species_colors=colors,
    )


def _compute_corridor_cost(
    selected_by_species: Mapping[str, list[PathCandidate]],
    cost_corridor_dict: Mapping[str, float],
) -> float:
    used_cells = set(
        cell
        for cands in selected_by_species.values()
        for cand in cands
        for cell in cand.cells
    )
    return sum(cost_corridor_dict.get(cell, 0.0) for cell in used_cells)


__all__ = [
    "MenorcaEcologicalCorridor",
    "PathCandidate",
    "generate_pairs",
    "enumerate_candidates",
    "build_cost_dicts",
    "report_selection",
    "collect_used_cells",
    "build_summary_from_paths",
    "_compute_corridor_cost",
]
