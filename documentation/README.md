# Wildlife Corridor Optimization for Menorca

This project implements Mixed-Integer Linear Programming (MILP) models for optimal wildlife corridor design and habitat adaptation in Menorca, a UNESCO Biosphere Reserve.

## Problem Overview

The goal is to design a network of wildlife corridors connecting habitat patches for four Mediterranean species while respecting ecological constraints and budget limitations:

- **Atelerix algirus** (North African hedgehog)
- **Martes martes** (European pine marten)  
- **Eliomys quercinus** (Garden dormouse)
- **Oryctolagus cuniculus** (European rabbit)

Key challenges:
- **Species incompatibility**: Pine marten (*Martes martes*) is a predator and cannot share corridors with rabbit and dormouse
- **Budget constraints**: Limited resources for corridor construction and habitat adaptation
- **Connectivity**: Origins of each species should be connected through the corridor network

## Available Models

| Model | Script | Description | Best For |
|-------|--------|-------------|----------|
| **Full MILP** | `modelling_multi_specie.py` | Complete multi-objective formulation | Small instances, exact solutions |
| **Distilled Heuristic** | `graph_model_gurobi_budgeted.py`  `graph_model_ortools_budgeted.py`  | 3-phase Algorithm (Prey → Predator → Rehab) | Large instances, practical use |
| **Less Heuristic Implementation** | `ortools_pipe/main.py` `gurobi_pipe_model_class.py` | 3-phase Algorithm (Atelerix → Predator & Prey → Rehab) | Medium instances with more flexibility |


## Quick Start

```bash
# Run the distilled heuristic model (recommended)
uv run python src/models/graph_model_gurobi_budgeted.py
uv run python src/models/graph_model_ortools_budgeted.py

# Run the full MILP model
uv run python src/models/modelling_multi_specie.py

# Run the less heuristic implementation
uv run python src/models/ortools_pipe/main.py
uv run python src/models/gurobi_pipe_model_class.py
```

## Documentation

- **[Multi-Species MILP Model](multi_species_model.md)**: Complete mathematical formulation with all decision variables, objective function, and constraints
- **[Distilled Heuristic (Algorithm 1)](distilled_model.md)**: Scalable 3-phase approach for large instances
- **[Running the Models](running_scripts.md)**: Command-line usage and configuration

## Project Structure

```
src/models/
├── modelling_multi_specie.py      # Full MILP model (Section II.C)
├── graph_model_gurobi_budgeted.py # Distilled heuristic (Algorithm 1)
├── graph_core.py                  # Graph utilities and path enumeration
├── visualization.py               # Map generation
├── gurobi/                        # Gurobi helper modules
│   ├── constants.py               # Default parameters
│   ├── variables.py               # Variable creation
│   └── constraints.py             # Constraint building
└── ortools_pipe/                  # OR-Tools implementation
    └── main.py
```

## Data Requirements

The models expect `data/processed_dataset.parquet` containing:
- `grid_id`: Unique cell identifier
- `geometry`: Cell polygon (WKB format)
- `cost_corridor`: Cost of building a corridor in this cell
- `has_{species}`: Boolean indicating species presence (origin cells)
- `cost_adaptation_{species}`: Adaptation cost per species
- `{species}_benefit`: Ecological benefit of adapting for each species

