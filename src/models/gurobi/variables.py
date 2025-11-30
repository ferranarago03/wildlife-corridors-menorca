"""
Reusable variable builders for the Gurobi corridor models.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from ..utils import manhattan_distance


def create_x_vars(model: gp.Model, all_cells: Sequence[str]) -> Dict[str, gp.Var]:
    return {j: model.addVar(vtype=GRB.BINARY, name=f"x_{j}") for j in all_cells}


def create_y_vars(
    model: gp.Model,
    species_list: Iterable[str],
    origin_cells_by_species: Mapping[str, Sequence[str]],
    all_cells: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
    *,
    max_distance: Optional[int] = None,
    far_cells_by_species: Optional[Mapping[str, Set[str]]] = None,
) -> Dict[Tuple[str, str, str, str], gp.Var]:
    y = {}
    far_cells_by_species = far_cells_by_species or {}
    for species in species_list:
        origins = origin_cells_by_species.get(species, [])
        far_cells = far_cells_by_species.get(species, set())
        for r in origins:
            for j in all_cells:
                if max_distance is not None and manhattan_distance(r, j) > max_distance:
                    continue
                if j in far_cells:
                    continue
                for k in adjacency[j]:
                    if k in far_cells:
                        continue
                    y[(species, r, j, k)] = model.addVar(
                        vtype=GRB.BINARY, name=f"y_{species}_{r}_{j}_{k}"
                    )
    return y


def create_u_vars(
    model: gp.Model, species_list: Iterable[str], all_cells: Sequence[str]
) -> Dict[Tuple[str, str], gp.Var]:
    u = {}
    for species in species_list:
        for j in all_cells:
            u[(species, j)] = model.addVar(vtype=GRB.BINARY, name=f"u_{species}_{j}")
    return u


def create_rehab_vars(
    model: gp.Model,
    species_list: Iterable[str],
    all_cells: Sequence[str],
    origin_cells_by_species: Mapping[str, Sequence[str]],
) -> Dict[Tuple[str, str], gp.Var]:
    rehab = {}
    for species in species_list:
        origin_cells = set(origin_cells_by_species.get(species, []))
        for j in all_cells:
            if j not in origin_cells:
                rehab[(species, j)] = model.addVar(
                    vtype=GRB.BINARY, name=f"rehab_{species}_{j}"
                )
    return rehab


__all__ = ["create_x_vars", "create_y_vars", "create_u_vars", "create_rehab_vars"]
