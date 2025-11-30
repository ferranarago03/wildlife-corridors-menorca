# Running the Models

## Prerequisites

```bash
# Install dependencies
uv sync

# Verify Gurobi installation (required for Gurobi models)
python -c "import gurobipy as gp; print(gp.gurobi.version())"

# Verify data availability
ls data/processed_dataset.parquet
```

## Quick Start

### Distilled Heuristic (Recommended)

```bash
uv run python src/models/graph_model_gurobi_budgeted.py
```

This runs Algorithm 1 with default settings:
- Budget: 500.0
- Time limit: 300s per phase
- Manhattan distance threshold: 15

### Full MILP Model

```bash
uv run python src/models/modelling_multi_specie.py
```

For small instances or validation. May timeout on full dataset.

### OR-Tools Implementation

```bash
uv run python src/models/ortools_pipe/main.py
```

No Gurobi license required.

## Configuration

### Modifying Parameters

Edit constants in `src/models/gurobi/constants.py`:

```python
BUDGET = 500.0           # Total budget
ALPHA = 0.5              # Cost-benefit balance (0=benefit, 1=cost)
TIME_LIMIT_METHOD = 300  # Solver time limit (seconds)
GAP = 0.2                # Optimality gap (0.2 = 20%)
```

### Per-Species Budget Allocation

In `graph_model_gurobi_budgeted.py`:

```python
CORRIDOR_SHARE_BY_SPECIES = {
    "oryctolagus_cuniculus": 0.30,
    "eliomys_quercinus": 0.24,
    "atelerix_algirus": 0.14,
    "martes_martes": 0.12,
}
```

## Output

### Generated Files

- **Maps**: Interactive HTML maps in `maps/` directory
  - `graph_model_gurobi_budgeted_map_all.html` - All species
  - `graph_model_gurobi_budgeted_map_{species}.html` - Per species

- **Summaries**: JSON solution summaries in `data/experiments/summaries/`

### Console Output

```
FASE 1: Corredores sin martes_martes
Grafo construido: 8234 nodos, 16102 aristas
Especie oryctolagus_cuniculus: 127 caminos de 156 pares evaluados
...
FASE 2: Corredores para martes_martes
...
FASE 3: Rehabilitación post-corredores
...
✓ Mapa general guardado en maps/graph_model_gurobi_budgeted_map_all.html
```

## Troubleshooting

### Gurobi License Issues

```bash
# Set license file path
export GRB_LICENSE_FILE="/path/to/gurobi.lic"

# Check license status
python -c "import gurobipy; m = gurobipy.Model(); print('License OK')"
```

### Out of Memory

For large instances:
1. Reduce `MAX_DISTANCE_METHOD` (limits path enumeration)
2. Increase `GAP` (allows earlier termination)
3. Use the distilled heuristic instead of full MILP

### No Solution Found

- Check if budget is sufficient: increase `BUDGET`
- Relax coverage requirements: set `MIN_COVERAGE_FRACTION = 0.5`
- Increase time limit: `TIME_LIMIT_METHOD = 600`
