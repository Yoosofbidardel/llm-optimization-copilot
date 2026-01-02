from typing import List

from api.schemas import Assignment


def build_explanation(status: str, assignments: List[Assignment], total_cost: float | None) -> str:
    if status != "OPTIMAL":
        return f"Solver finished with status {status}. No feasible assignment found."
    lines = [f"Optimised assignment with total cost {total_cost}:"]
    for assign in assignments:
        if assign.resource_id is None:
            lines.append(f"- {assign.task_id} was left unassigned.")
        else:
            lines.append(f"- {assign.task_id} -> {assign.resource_id} (cost={assign.cost})")
    return "\n".join(lines)
