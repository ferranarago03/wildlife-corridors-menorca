# Distilled Heuristic Model (Algorithm 1)

**Script**: `src/models/graph_model_gurobi_budgeted.py`

This document describes the distilled heuristic approach that enables practical optimization on large landscapes. It corresponds to **Section II.D** of the project report.

## Motivation

The [full MILP model](multi_species_model.md) suffers from a major scalability problem: with $O(|S| \cdot |O| \cdot |C|^2)$ flow variables, it becomes computationally intractable for the complete Menorca dataset. The distilled heuristic addresses this by:

1. **Decomposing** the problem into sequential phases
2. **Using Dijkstra's algorithm** to enumerate candidate paths (instead of flow variables)
3. **Solving smaller ILPs** to select optimal path subsets

## Algorithm Overview

The algorithm divides species into two groups based on predator-prey relationships:

- **Prey species**: Oryctolagus cuniculus, Eliomys quercinus, Atelerix algirus
- **Predator species**: Martes martes

```
Algorithm 1: Multi-species Ecological Corridor Design

Phase 1: Optimize corridors for prey species
    - Build graph G (excluding martes-only origins)
    - Enumerate shortest paths between prey origins using Dijkstra
    - Solve ILP to select minimum-cost path cover

Phase 2: Optimize corridors for predator species  
    - Build graph G' = G minus cells used by prey in Phase 1
    - Enumerate shortest paths between martes origins
    - Solve ILP to select minimum-cost path cover

Phase 3: Habitat rehabilitation
    - Identify candidate cells adjacent to built corridors
    - Solve ILP maximizing benefit minus cost
    - Enforce predator-prey compatibility constraints
```

## Phase Details

### Phase 1: Non-Conflicting Species (Prey)

**Input**: Full landscape graph $G$, species set $S_{prey} = S \setminus \{\text{Martes}\}$

**Process**:
1. Generate origin pairs $(o_i, o_j)$ for each prey species where $\text{manhattan}(o_i, o_j) \leq D_{max}$
2. Compute shortest path for each pair using Dijkstra's algorithm
3. Create `PathCandidate` objects with path cells and costs
4. Solve path selection ILP:

$$\min \sum_{p \in P} \text{cost}_p \cdot z_p$$

Subject to:
- Each origin participates in at least one selected path
- Per-species budget constraints

**Output**: Selected paths for prey species, cells used

### Phase 2: Conflicting Species (Predator)

**Input**: Graph $G'$ with incompatible cells removed

**Incompatible cells** (removed from $G'$):
- Origin cells of Oryctolagus and Eliomys unless origins.
- Interior cells of paths selected in Phase 1 for Oryctolagus and Eliomys

**Process**: Same as Phase 1, but only for Martes martes

**Output**: Selected paths for Martes

### Phase 3: Rehabilitation

**Input**: Fixed corridor network from Phases 1 & 2

**Candidate cells**: Cells adjacent to built corridors or origins

**Objective**:
$$\min \alpha \sum_{s,j} a_{sj} R_{sj} - (1-\alpha) \sum_{s,j} b_{sj} R_{sj}$$

**Constraints**:
- Compatibility: $2R_{\text{martes},j} + R_{\text{oryctolagus},j} + R_{\text{eliomys},j} \leq 2$
- Per-species adaptation budgets

## Implementation

### Code Structure

```python
# graph_model_gurobi_budgeted.py

def run_pipeline():
    # Phase 1: Prey species
    graph_phase1 = CorredorEcologicoMenorca(...)
    graph_phase1.construir_grafo(excluded_cells=martes_only_origins)
    
    candidates_phase1, _, origin_to_paths_p1 = enumerate_candidates(
        graph_phase1, without_martes, MAX_DISTANCE
    )
    solution_phase1 = solve_path_selection(candidates_phase1, ...)
    
    # Phase 2: Martes (on pruned graph)
    cells_to_remove = collect_incompatible_cells(solution_phase1)
    graph_phase2 = CorredorEcologicoMenorca(...)
    graph_phase2.construir_grafo(excluded_cells=cells_to_remove)
    
    candidates_martes, _, origin_to_paths_martes = enumerate_candidates(
        graph_phase2, ["martes_martes"], MAX_DISTANCE
    )
    solution_martes = solve_path_selection(candidates_martes, ...)
    
    # Phase 3: Rehabilitation
    used_cells = collect_used_cells(solution_phase1, solution_martes)
    rehab_selected = solve_rehab(used_cells, ...)
```

### Core Components

**`CorredorEcologicoMenorca`** (in `graph_core.py`):
- Builds NetworkX graph from grid adjacency
- Computes shortest paths using Dijkstra
- Supports cell exclusion for phased optimization

**`PathCandidate`** dataclass:
```python
@dataclass
class PathCandidate:
    path_id: str      # Unique identifier
    species: str      # Species name
    origin: str       # Starting cell
    destination: str  # Ending cell
    cells: list[str]  # All cells in path
    cost: float       # Sum of corridor costs (excluding origin)
    length: int       # Number of edges
```

**`solve_path_selection`**: ILP for selecting minimum-cost path cover with coverage and budget constraints

**`solve_rehab`**: ILP for rehabilitation optimization with compatibility constraints

### Configuration

Key parameters:

```python
TOTAL_BUDGET = 500.0
MAX_DISTANCE_METHOD = 15  # Maximum Manhattan distance for pair generation

# Per-species budget allocation
CORRIDOR_SHARE_BY_SPECIES = {
    "oryctolagus_cuniculus": 0.30,
    "eliomys_quercinus": 0.24,
    "atelerix_algirus": 0.14,
    "martes_martes": 0.12,
}
ADAPTATION_SHARE_BY_SPECIES = {
    "oryctolagus_cuniculus": 0.07,
    "eliomys_quercinus": 0.06,
    "atelerix_algirus": 0.04,
    "martes_martes": 0.03,
}
```

## Limitations

- **Heuristic nature**: The sequential phasing may miss globally optimal solutions where a predator detour enables cheaper prey paths
- **Path enumeration**: Only considers shortest paths; alternative routes are not explored
- **Fixed ordering**: Prey-first strategy is hardcoded; alternative orderings not explored

## Comparison with 3-Phase Pipeline

The distilled heuristic differs from `gurobi_pipe_model_class.py`:

| Feature | Distilled Heuristic | 3-Phase Pipeline |
|---------|---------------------|------------------|
| Phase 1 species | All prey (Oryctolagus, Eliomys, Atelerix) | Atelerix only |
| Phase 2 species | Martes only | All others |
| Conflict handling | Explicit predator-prey separation | Mixed approach |
| Recommended use | Production / large instances | Experimental |
