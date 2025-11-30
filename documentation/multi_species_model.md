# Multi-Species MILP Model

**Script**: `src/models/modelling_multi_specie.py`

This document describes the complete Mixed-Integer Linear Programming formulation for multi-species wildlife corridor optimization. It corresponds to **Section II** of the project report.

## Problem Statement

Given a discretized landscape grid of Menorca, the model simultaneously optimizes:
1. **Corridor construction**: Selecting cells to build wildlife corridors
2. **Habitat adaptation**: Rehabilitating cells adjacent to corridors for species-specific needs
3. **Species connectivity**: Ensuring habitat patches (origins) are connected for each species

The model handles predator-prey incompatibility between *Martes martes* (predator) and *Oryctolagus cuniculus* / *Eliomys quercinus* (prey).

## Mathematical Formulation

### Sets and Parameters

| Symbol | Description |
|--------|-------------|
| $i, j, k$ | Indices of grid cells |
| $s$ | Species ∈ {atelerix algirus, martes martes, eliomys quercinus, oryctolagus cuniculus} |
| $c_j$ | Cost of building a corridor in cell $j$ |
| $a_{sj}$ | Cost of adapting cell $j$ for species $s$ |
| $b_{sj}$ | Benefit of adapting cell $j$ for species $s$ |
| $A_j$ | Set of cells adjacent to cell $j$ |
| $r_s$ | Set of origin cells for species $s$ |
| $p$ | Penalty for unconnected origins |
| $B_s^c$ | Budget for corridors for species $s$ |
| $B_s^a$ | Budget for adaptation for species $s$ |

### Decision Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| $X_j$ | {0,1} | 1 if a corridor is built in cell $j$ |
| $Y_{rsjk}$ | {0,1} | 1 if there is flow from origin $r$ through arc $(j,k)$ for species $s$ |
| $U_{sj}$ | {0,1} | 1 if species $s$ uses cell $j$ as a corridor |
| $R_{sj}$ | {0,1} | 1 if cell $j$ is rehabilitated for species $s$ |
| $C_{rs}$ | {0,1} | 1 if origin $r$ of species $s$ is connected |

### Objective Function

The model uses α-lexicographic optimization balancing costs and benefits:

$$\min z = \alpha \left( \sum_j c_j X_j + \sum_{s,j} a_{sj} R_{sj} + p \sum_{r,s} (1 - C_{rs}) \right) - (1-\alpha) \sum_{s,j} b_{sj} R_{sj}$$

Where $\alpha = 0.5$ balances corridor/adaptation costs against ecological benefits.

### Constraints

#### Origin Connectivity (Equations 2-3)

Each origin must have at least one outgoing flow if connected:

$$\sum_{k \in A_r} Y_{rsrk} \leq |A_r| \cdot C_{rs} \quad \forall r, s$$

$$Y_{rsjk} \leq C_{rs} \quad \forall r, s, j, k \in A_j$$

#### Flow Conservation (Equation 4)

Flow entering a non-origin cell must equal flow leaving:

$$\sum_{i \in A_j} Y_{rsij} - \sum_{k \in A_j} Y_{rsjk} = 0 \quad \forall s, r, j \notin r_s$$

#### No Reverse Flow (Equation 5)

Prevents bidirectional flow on the same edge:

$$Y_{rsjk} + Y_{rskj} \leq 1 \quad \forall r_s, j, k \in A_j \text{ with } j < k$$

#### Flow Activation (Equation 6)

Links flow variables to corridor usage:

$$\sum_{r,k} (Y_{rsjk} + Y_{rskj}) \leq M \cdot U_{sj} \quad \forall s, j$$

$$U_{sj} \leq \sum_{r,k} (Y_{rsjk} + Y_{rskj}) \quad \forall s, j$$

Where $M$ is the maximum number of incident arcs.

#### Corridor Construction Link (Equation 7)

Species usage requires corridor construction:

$$\sum_s U_{sj} \leq |S| \cdot X_j \quad \forall j$$

#### Species Incompatibility (Equation 8)

Predator cannot share corridors with prey (except at shared origins):

$$2 U_{\text{martes},j} + U_{\text{oryctolagus},j} + U_{\text{eliomys},j} \leq 2 \quad \forall j$$

#### Rehabilitation Adjacency (Equation 9)

A cell can only be adapted if adjacent to a corridor or origin:

$$R_{sj} \leq \sum_{k \in A_j} U_{sk} + m \quad \forall s, j \notin r_s$$

Where $m = 1$ if any neighbor is an origin, 0 otherwise.

#### Rehabilitation Compatibility (Equations 10-12)

Mirrors corridor incompatibility for adapted cells:

$$2 R_{\text{martes},j} + R_{\text{oryctolagus},j} + R_{\text{eliomys},j} \leq 2 \quad \forall j$$

$$2 U_{\text{martes},j} + R_{\text{oryctolagus},j} + R_{\text{eliomys},j} \leq 2 \quad \forall j$$

$$2 R_{\text{martes},j} + U_{\text{oryctolagus},j} + U_{\text{eliomys},j} \leq 2 \quad \forall j$$

#### Budget Constraints (Equations 13-14)

Per-species corridor and adaptation budgets:

$$\sum_j c_j U_{sj} \leq B_s^c \quad \forall s$$

$$\sum_j a_{sj} R_{sj} \leq B_s^a \quad \forall s$$

#### Minimum Coverage (Equation 15)

Optional constraint for minimum origin connectivity:

$$\sum_{r,s} C_{rs} \geq \text{cover} \cdot |r_s|$$

## Implementation

### Code Structure

The model is implemented in `modelling_multi_specie.py` with the following structure:

```python
# 1. Data loading
df = gpd.read_parquet("data/processed_dataset.parquet")

# 2. Build adjacency graph
adjacency = {cell: get_adjacent_cells(cell, all_cells_set, df) for cell in all_cells}

# 3. Create Gurobi model and variables
model = gp.Model("WildlifeCorridors")
x = {j: model.addVar(vtype=GRB.BINARY) for j in all_cells}
y = {}  # Flow variables indexed by (species, origin, j, k)
u = {}  # Usage variables indexed by (species, j)
rehab = {}  # Rehabilitation variables

# 4. Set objective and add constraints
model.setObjective(alpha * cost - (1-alpha) * benefit, GRB.MINIMIZE)

# 5. Solve and extract solution
model.optimize()
```

### Key Design Decisions

1. **Flow-based connectivity**: Uses network flow formulation rather than spanning trees, allowing partial connectivity when budget is insufficient

2. **Species-specific budgets**: Budget is split per species (configurable via `CORRIDOR_SHARE_BY_SPECIES` and `ADAPTATION_SHARE_BY_SPECIES`)

3. **Optional coverage**: Uses penalty-based soft constraints for origin coverage, allowing solutions even when full connectivity is infeasible

4. **Big-M values**: Derived from actual adjacency counts to keep LP relaxation tight

### Configuration

Key parameters in `src/models/gurobi/constants.py`:

```python
SPECIES = ["oryctolagus_cuniculus", "atelerix_algirus", 
           "eliomys_quercinus", "martes_martes"]
BUDGET = 500.0
ALPHA = 0.5
TIME_LIMIT_METHOD = 300  # seconds
GAP = 0.2  # 20% optimality gap
```

## Limitations

- **Scalability**: The full model has $O(|S| \cdot |O| \cdot |C|^2)$ flow variables, making it impractical for large grids (> 5000 cells)
- **Memory**: Requires significant RAM for the constraint matrix
- **Solution time**: May not converge within reasonable time limits on the full Menorca dataset

For large instances, use the [Distilled Heuristic (Algorithm 1)](distilled_model.md) instead.
