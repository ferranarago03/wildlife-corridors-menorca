"""
Helper to run `models/modelling_multi_specie.py` with different parameter sets
without editing the model file.

- Accepts a JSON config with a list of runs or a single run via CLI flags.
- Patches the constants module (ALPHA, BUDGET, GAP, etc.) for each run.
- Optionally overrides the corridor/adaptation shares to change per-species budgets.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import models.gurobi.constants as constants

MODEL_PATH = SRC / "models" / "modelling_multi_specie.py"
BASE_SOURCE = MODEL_PATH.read_text()

PARAM_MAP = {
    "alpha": "ALPHA",
    "budget": "BUDGET",
    "gap": "GAP",
    "focus": "FOCUS",
    "heuristics": "HEURISTICS",
    "penalty_uncovered_origin": "PENALTY_UNCOVERED_ORIGIN",
    "min_coverage_fraction": "MIN_COVERAGE_FRACTION",
}

DEFAULT_RUNS = [
    {
        "name": "baseline",
        "alpha": constants.ALPHA,
        "budget": constants.BUDGET,
        "gap": constants.GAP,
    },
    {
        "name": "low_alpha",
        "alpha": 0.3,
        "budget": constants.BUDGET,
        "gap": constants.GAP,
    },
    {
        "name": "high_budget",
        "alpha": constants.ALPHA,
        "budget": constants.BUDGET * 1.5,
        "gap": constants.GAP,
    },
]


def _parse_dict_arg(raw: str | None) -> dict[str, float] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("Value must be a JSON dictionary {'species': share}.")
    return {str(k): float(v) for k, v in parsed.items()}


def _load_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.config:
        data = json.loads(pathlib.Path(args.config).read_text())
        if isinstance(data, dict) and "runs" in data:
            runs = data["runs"]
        elif isinstance(data, list):
            runs = data
        else:
            raise SystemExit("JSON must be a list of runs or contain a 'runs' key.")
        return runs

    single_values = any(
        getattr(args, key) is not None
        for key in [
            "alpha",
            "budget",
            "gap",
            "focus",
            "heuristics",
            "penalty_uncovered_origin",
            "min_coverage_fraction",
            "corridor_shares",
            "adaptation_shares",
        ]
    )
    if single_values:
        return [
            {
                "name": args.name,
                "alpha": args.alpha,
                "budget": args.budget,
                "gap": args.gap,
                "focus": args.focus,
                "heuristics": args.heuristics,
                "penalty_uncovered_origin": args.penalty_uncovered_origin,
                "min_coverage_fraction": args.min_coverage_fraction,
                "corridor_share_by_species": _parse_dict_arg(args.corridor_shares),
                "adaptation_share_by_species": _parse_dict_arg(args.adaptation_shares),
            }
        ]

    return DEFAULT_RUNS


def _patch_constants(overrides: dict[str, Any]) -> dict[str, Any]:
    original: dict[str, Any] = {}
    for key, attr in PARAM_MAP.items():
        if key in overrides and overrides[key] is not None:
            original[attr] = getattr(constants, attr)
            setattr(constants, attr, overrides[key])
    return original


def _restore_constants(original: dict[str, Any]) -> None:
    for attr, value in original.items():
        setattr(constants, attr, value)


def _inject_shares(
    source: str,
    corridor_shares: dict[str, float] | None,
    adaptation_shares: dict[str, float] | None,
) -> str:
    patched = source
    if corridor_shares is not None:
        replacement = (
            "CORRIDOR_SHARE_BY_SPECIES: dict[str, float] = "
            f"{json.dumps(corridor_shares, indent=4, sort_keys=True)}\n"
        )
        patched, count = re.subn(
            r"CORRIDOR_SHARE_BY_SPECIES: dict\[str, float\]\s*=\s*\{.*?\}\n",
            replacement,
            patched,
            count=1,
            flags=re.DOTALL,
        )
        if count == 0:
            raise RuntimeError("Could not patch CORRIDOR_SHARE_BY_SPECIES.")

    if adaptation_shares is not None:
        replacement = f"ADAPTATION_SHARE_BY_SPECIES: dict[str, float] | None = {repr(adaptation_shares)}"
        patched, count = re.subn(
            r"ADAPTATION_SHARE_BY_SPECIES: dict\[str, float\] \| None = .*",
            replacement,
            patched,
            count=1,
        )
        if count == 0:
            raise RuntimeError("Could not patch ADAPTATION_SHARE_BY_SPECIES.")

    return patched


def run_single(run_cfg: dict[str, Any]) -> None:
    name = run_cfg.get("name", "run")
    print(f"\n=== Running experiment: {name} ===")
    original = _patch_constants(run_cfg)
    try:
        source = _inject_shares(
            BASE_SOURCE,
            run_cfg.get("corridor_share_by_species"),
            run_cfg.get("adaptation_share_by_species"),
        )
        run_globals = {"__name__": "__main__", "__file__": str(MODEL_PATH)}
        exec(compile(source, str(MODEL_PATH), "exec"), run_globals)
    finally:
        _restore_constants(original)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run modelling_multi_specie.py with different parameters."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON with list of runs. Accepts list or {'runs': [...]}.",
    )
    parser.add_argument(
        "--name", type=str, default="manual", help="Name for a single run."
    )
    parser.add_argument("--alpha", type=float, help="Objective ALPHA.")
    parser.add_argument("--budget", type=float, help="Total BUDGET.")
    parser.add_argument("--gap", type=float, help="MIP Gap.")
    parser.add_argument("--focus", type=int, help="MIP Focus.")
    parser.add_argument("--heuristics", type=float, help="Heuristics.")
    parser.add_argument(
        "--penalty-uncovered-origin",
        type=float,
        help="Penalty for uncovered origin.",
    )
    parser.add_argument(
        "--min-coverage-fraction", type=float, help="Minimum origin coverage fraction."
    )
    parser.add_argument(
        "--corridor-shares",
        type=str,
        help="JSON dict of shares per species for corridors (e.g. '{\"oryctolagus_cuniculus\":0.4}')",
    )
    parser.add_argument(
        "--adaptation-shares",
        type=str,
        help="JSON dict of shares per species for adaptation or null.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = _load_runs(args)
    for run_cfg in runs:
        run_single(run_cfg)


if __name__ == "__main__":
    main()
