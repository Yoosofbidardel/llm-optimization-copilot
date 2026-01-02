from typing import List

from ortools.linear_solver import pywraplp

from api.schemas import Assignment, ProblemSpec, Solution
from solver.explain import build_explanation
from solver.translators import build_capacities, build_cost_matrix


def solve_problem(spec: ProblemSpec) -> Solution:
    task_ids, resource_ids, cost_matrix = build_cost_matrix(spec)
    capacities = build_capacities(spec)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("Failed to create OR-Tools solver; ensure ortools is installed.")

    x = {}
    for i, task in enumerate(task_ids):
        for j, res in enumerate(resource_ids):
            x[(i, j)] = solver.BoolVar(f"x_{task}_{res}")

    # Each task assigned at most once, or exactly once if required
    for i, task in enumerate(task_ids):
        constraint = solver.Sum([x[(i, j)] for j in range(len(resource_ids))])
        if spec.must_assign_all_tasks:
            solver.Add(constraint == 1)
        else:
            solver.Add(constraint <= 1)

    # Capacity constraints per resource
    for j, res in enumerate(resource_ids):
        solver.Add(
            solver.Sum([x[(i, j)] for i in range(len(task_ids))]) <= capacities.get(res, 0)
        )

    # Objective: minimise cost
    objective_terms = []
    for i in range(len(task_ids)):
        for j in range(len(resource_ids)):
            objective_terms.append(cost_matrix[i][j] * x[(i, j)])
    solver.Minimize(solver.Sum(objective_terms))

    status = solver.Solve()
    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    }
    status_text = status_map.get(status, "UNKNOWN")

    assignments: List[Assignment] = []
    total_cost = None
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        total_cost = solver.Objective().Value()
        for i, task in enumerate(task_ids):
            assigned = False
            for j, res in enumerate(resource_ids):
                if x[(i, j)].solution_value() > 0.5:
                    assignments.append(
                        Assignment(
                            task_id=task,
                            resource_id=res,
                            cost=cost_matrix[i][j],
                        )
                    )
                    assigned = True
                    break
            if not assigned:
                assignments.append(Assignment(task_id=task, resource_id=None, cost=None))

    explanation = build_explanation(status_text, assignments, total_cost)
    return Solution(
        status=status_text,
        total_cost=total_cost,
        assignments=assignments,
        explanation=explanation,
    )
