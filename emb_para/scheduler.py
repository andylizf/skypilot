from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator import DynamicSchedulerSimulator
    from simulator import Cluster
    from cluster import Task

class Scheduler(ABC):
    """Abstract base class for scheduling algorithms.
    All custom scheduling algorithms must inherit from this class.
    """

    @abstractmethod
    def __init__(self, cluster_config: 'Cluster'):
        """Initialize the scheduler with cluster configuration."""
        pass

    @abstractmethod
    def schedule_tasks(self, simulator: 'DynamicSchedulerSimulator', new_tasks: list['Task']):
        """Schedule tasks and manage clusters."""
        pass
