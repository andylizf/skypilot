from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    """Represents a task in the simulation."""
    task_id: int
    arrival_time: int
    execution_time: int = 5
    gpu_requirement: int = 1
    start_time: Optional[int] = None
    end_time: Optional[int] = None


@dataclass
class ClusterConfig:
    """Configuration for cluster instances."""
    gpu_count: int
    startup_time: int
    cost_per_minute: float


class Cluster:
    """Represents a cluster with a fixed number of GPUs."""
    def __init__(self, cluster_id: int, config: ClusterConfig, start_time: int):
        self.cluster_id = cluster_id
        self.config = config
        self.active_tasks: list[Task] = []
        self.start_time = start_time
        self.shutdown_time: Optional[int] = None
        self.ready_time = start_time + config.startup_time

    def is_available(self):
        """Check if the cluster can accept more tasks."""
        remaining_gpus = self.config.gpu_count - sum(task.gpu_requirement for task in self.active_tasks)
        return remaining_gpus > 0

    def can_accommodate(self, task: Task):
        """Check if the cluster can accommodate a specific task."""
        remaining_gpus = self.config.gpu_count - sum(task.gpu_requirement for task in self.active_tasks)
        return remaining_gpus >= task.gpu_requirement

    def start_tasks(self, task: Task,current_time: int):
        assert self.ready_time <= current_time
        assert self.is_available()
        self.active_tasks.append(task)
        task.start_time = current_time
        task.end_time = task.start_time + task.execution_time
        print(f"    Current GPU usage: {len(self.active_tasks)}/{self.config.gpu_count}")

    def shutdown(self, current_time: int):
        """Shutdown the cluster."""
        assert not self.shutdown_time
        assert len(self.active_tasks) == 0
        self.shutdown_time = current_time
