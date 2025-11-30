from __future__ import annotations

import ortools.linear_solver.pywraplp as pywraplp

from .base_lp import ORToolsPipeModelBase


class SCIPPipeModel(ORToolsPipeModelBase):
    """Pipeline implementation using the SCIP solver."""

    def _create_solver(self):
        return pywraplp.Solver.CreateSolver("SCIP")

    def set_solver_parameters(
        self,
        solver: pywraplp.Solver,
        num_threads: int,
        time_limit_ms: int,
        gap: float = 0.2,
    ):
        solver.SetNumThreads(num_threads)
        solver.SetTimeLimit(time_limit_ms)

        params_string = f"limits/gap={gap}"
        solver.SetSolverSpecificParametersAsString(params_string)

        solver.EnableOutput()


class CBCPipeModel(ORToolsPipeModelBase):
    """Pipeline implementation using the CBC solver."""

    def _create_solver(self):
        return pywraplp.Solver.CreateSolver("CBC")

    def set_solver_parameters(
        self,
        solver: pywraplp.Solver,
        num_threads: int,
        time_limit_ms: int,
        gap: float = 0.2,
    ):
        solver.SetNumThreads(num_threads)
        solver.SetTimeLimit(time_limit_ms)

        params_string = f"ratioGap={gap}"
        solver.SetSolverSpecificParametersAsString(params_string)

        solver.EnableOutput()


class GLPKPipeModel(ORToolsPipeModelBase):
    """Pipeline implementation using the GLPK solver."""

    def _create_solver(self):
        return pywraplp.Solver.CreateSolver("GLPK_MIXED_INTEGER_PROGRAMMING")

    def set_solver_parameters(
        self,
        solver: pywraplp.Solver,
        num_threads: int,
        time_limit_ms: int,
        gap: float = 0.2,
    ):
        solver.SetNumThreads(num_threads)
        solver.SetTimeLimit(time_limit_ms)

        params_string = f"mip_gap={gap}"
        solver.SetSolverSpecificParametersAsString(params_string)

        solver.EnableOutput()
