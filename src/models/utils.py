from __future__ import annotations

import geopandas as gpd


def parse_grid_id(grid_id: str) -> tuple[int, int]:
    """Extract row and column from grid_id format 'cell_row_col'."""
    parts = grid_id.split("_")
    return int(parts[1]), int(parts[2])


def manhattan_distance(cell_a: str, cell_b: str) -> int:
    """Calculate Manhattan distance between two grid cells."""
    row_a, col_a = parse_grid_id(cell_a)
    row_b, col_b = parse_grid_id(cell_b)
    return abs(row_a - row_b) + abs(col_a - col_b)


def get_adjacent_cells(
    grid_id: str, all_cells_set: set[str], df: gpd.GeoDataFrame
) -> list[str]:
    """
    Get adjacent cells in the four cardinal directions if they touch the target cell.
    """
    row, col = parse_grid_id(grid_id)
    adjacent: list[str] = []
    geom_cell = df[df["grid_id"] == grid_id]["geometry"].values[0]

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor_id = f"cell_{row + dr}_{col + dc}"
        if neighbor_id in all_cells_set:
            geom_neighbor = df[df["grid_id"] == neighbor_id]["geometry"].values[0]
            if geom_cell.touches(geom_neighbor):
                adjacent.append(neighbor_id)
    return adjacent
