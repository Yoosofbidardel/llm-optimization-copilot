from typing import Dict, List, Tuple

from api.schemas import ProblemSpec


def build_cost_matrix(spec: ProblemSpec) -> Tuple[List[str], List[str], List[List[float]]]:
    task_ids = [task.id for task in spec.tasks]
    resource_ids = [res.id for res in spec.resources]
    cost_matrix: List[List[float]] = []
    for task in spec.tasks:
        row = []
        for res_id in resource_ids:
            row.append(task.cost_per_resource.get(res_id, float("inf")))
        cost_matrix.append(row)
    return task_ids, resource_ids, cost_matrix


def build_capacities(spec: ProblemSpec) -> Dict[str, int]:
    return {res.id: res.capacity for res in spec.resources}
