"""
Constraint blocks shared by the Gurobi corridor models.
Each helper is small and self-contained to keep the main scripts readable.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import gurobipy as gp


def add_flow_constraints(
    model: gp.Model,
    y: Dict[Tuple[str, str, str, str], gp.Var],
    species_list: Iterable[str],
    origin_cells_by_species: Mapping[str, Sequence[str]],
    all_cells: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
):
    for species in species_list:
        origins = origin_cells_by_species[species]
        for r in origins:
            outflow = gp.LinExpr()
            for j in adjacency[r]:
                if (species, r, r, j) in y:
                    outflow += y[(species, r, r, j)]
            model.addConstr(outflow >= 1, name=f"origin_outflow_{species}_{r}")
        for j in all_cells:
            for r in origins:
                if j in origins:
                    continue
                balance = gp.LinExpr()
                for i in adjacency[j]:
                    if (species, r, i, j) in y:
                        balance += y[(species, r, i, j)]
                for k in adjacency[j]:
                    if k != r and (species, r, j, k) in y:
                        balance -= y[(species, r, j, k)]
                model.addConstr(balance == 0, name=f"flow_{species}_{r}_{j}")


def add_no_reverse_flow_constraints(
    model: gp.Model,
    y: Dict[Tuple[str, str, str, str], gp.Var],
    species_list: Iterable[str],
    origin_cells_by_species: Mapping[str, Sequence[str]],
    all_cells: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
):
    for species in species_list:
        origins = origin_cells_by_species[species]
        for r in origins:
            for j in all_cells:
                for k in adjacency[j]:
                    if j < k and (species, r, j, k) in y and (species, r, k, j) in y:
                        model.addConstr(
                            y[(species, r, j, k)] + y[(species, r, k, j)] <= 1,
                            name=f"no_reverse_flow_{species}_{r}_{j}_{k}",
                        )


def add_activation_constraints(
    model: gp.Model,
    y: Dict[Tuple[str, str, str, str], gp.Var],
    u: Dict[Tuple[str, str], gp.Var],
    species_list: Iterable[str],
    origin_cells_by_species: Mapping[str, Sequence[str]],
    all_cells: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
):
    for species in species_list:
        origins = origin_cells_by_species[species]
        for j in all_cells:
            usage = gp.LinExpr()
            count_edges = 0
            for r in origins:
                for i in adjacency[j]:
                    if (species, r, i, j) in y:
                        usage += y[(species, r, i, j)]
                        count_edges += 1
                for k in adjacency[j]:
                    if (species, r, j, k) in y:
                        usage += y[(species, r, j, k)]
                        count_edges += 1
            M_cell = max(1, count_edges)
            if (species, j) in u:
                model.addConstr(
                    usage <= M_cell * u[(species, j)], name=f"use_up_{species}_{j}"
                )
                model.addConstr(u[(species, j)] <= usage, name=f"use_lo_{species}_{j}")


def add_link_u_x_single_species(
    model: gp.Model,
    u: Dict[Tuple[str, str], gp.Var],
    x: Dict[str, gp.Var],
    species: str,
):
    for j in x:
        model.addConstr(u[(species, j)] <= x[j], name=f"link_u_x_{species}_{j}")


def add_link_u_x_all_species(
    model: gp.Model,
    u: Dict[Tuple[str, str], gp.Var],
    x: Dict[str, gp.Var],
    species_list: Iterable[str],
):
    species_list = list(species_list)
    for j in x:
        lhs = gp.LinExpr()
        for species in species_list:
            if (species, j) in u:
                lhs += u[(species, j)]
        model.addConstr(lhs <= len(species_list) * x[j], name=f"link_u_x_all_{j}")


def add_species_compatibility(
    model: gp.Model,
    u: Dict[Tuple[str, str], gp.Var],
    all_cells: Sequence[str],
    origin_cells_by_species: Mapping[str, Sequence[str]],
):
    martes_orig = origin_cells_by_species.get("martes_martes", [])
    oryc_orig = origin_cells_by_species.get("oryctolagus_cuniculus", [])
    elio_orig = origin_cells_by_species.get("eliomys_quercinus", [])
    for j in all_cells:
        if j in martes_orig and (j in oryc_orig or j in elio_orig):
            continue
        model.addConstr(
            2 * u[("martes_martes", j)]
            + u[("oryctolagus_cuniculus", j)]
            + u[("eliomys_quercinus", j)]
            <= 2,
            name=f"species_compatibility_{j}",
        )


def add_rehab_constraints(
    model: gp.Model,
    rehab: Dict[Tuple[str, str], gp.Var],
    u: Dict[Tuple[str, str], gp.Var],
    all_cells: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
):
    for (species, j), var in rehab.items():
        adj_usage = gp.LinExpr()
        for k in adjacency[j]:
            if (species, k) in u:
                adj_usage += u[(species, k)]
        model.addConstr(var <= adj_usage, name=f"rehab_adjacent_corridor_{species}_{j}")
    for j in all_cells:
        model.addConstr(
            2 * rehab.get(("martes_martes", j), 0)
            + rehab.get(("oryctolagus_cuniculus", j), 0)
            + rehab.get(("eliomys_quercinus", j), 0)
            <= 2,
            name=f"rehab_compatibility_{j}",
        )


__all__ = [
    "add_activation_constraints",
    "add_flow_constraints",
    "add_link_u_x_all_species",
    "add_link_u_x_single_species",
    "add_no_reverse_flow_constraints",
    "add_rehab_constraints",
    "add_species_compatibility",
]
