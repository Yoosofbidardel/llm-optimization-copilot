from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TaskSpec(BaseModel):
    id: str = Field(..., description="Unique task identifier")
    cost_per_resource: Dict[str, float] = Field(
        ..., description="Cost of assigning the task to each resource"
    )

    @field_validator("cost_per_resource")
    @classmethod
    def validate_costs(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("cost_per_resource must include at least one resource entry")
        return value


class ResourceSpec(BaseModel):
    id: str = Field(..., description="Resource identifier")
    capacity: int = Field(1, ge=1, description="Maximum number of tasks the resource can take")


class ProblemSpec(BaseModel):
    tasks: List[TaskSpec]
    resources: List[ResourceSpec]
    must_assign_all_tasks: bool = Field(
        True,
        description="If true, every task must be assigned to exactly one resource; otherwise tasks may remain unassigned",
    )

    @field_validator("tasks", "resources")
    @classmethod
    def non_empty(cls, value, field):
        if not value:
            raise ValueError(f"{field.alias} must not be empty")
        return value


class Assignment(BaseModel):
    task_id: str
    resource_id: Optional[str]
    cost: Optional[float]


class Solution(BaseModel):
    status: str
    total_cost: Optional[float]
    assignments: List[Assignment]
    explanation: str
