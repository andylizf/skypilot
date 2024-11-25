import math
from typing import TYPE_CHECKING
from emb_para.scheduler import Scheduler

if TYPE_CHECKING:
    from emb_para.simulator import DynamicSchedulerSimulator
    from emb_para.cluster import Task

class GreedyScheduler(Scheduler):
    """A scheduler that dynamically manages cluster instances based on demand."""
    def __init__(self):
        self.pending_tasks: list['Task'] = []

    def schedule_tasks(self, simulator: 'DynamicSchedulerSimulator', new_tasks: list['Task']):
        """Assign tasks to clusters based on a greedy algorithm."""
        tasks = self.pending_tasks + new_tasks
        self.pending_tasks = []
        for task in tasks:
            for cluster in simulator.clusters:
                if (not cluster.shutdown_time and
                    cluster.can_accommodate(task) and
                        cluster.ready_time <= simulator.current_time):
                        print(f"  → Assigned task {task.task_id} to existing cluster {cluster.cluster_id}")
                        cluster.start_tasks(task, simulator.current_time)
                        break   
            else:
                self.pending_tasks.append(task)

        if self.pending_tasks:
            # Assume all pending tasks have the same GPU requirement
            task_per_cluster = simulator.cluster_config.gpu_count / self.pending_tasks[0].gpu_requirement
            need_clusters = math.ceil(len(self.pending_tasks) / task_per_cluster)
            starting_clusters = [
                cluster for cluster in simulator.clusters if cluster.ready_time > simulator.current_time
            ]
            for _ in range(need_clusters - len(starting_clusters)):
                new_cluster = simulator.create_cluster()
                print(f'  → Created new cluster {new_cluster.cluster_id}')
                print(f'    Will ready at time {new_cluster.ready_time}')

        for cluster in list(simulator.clusters):
            if len(cluster.active_tasks) == 0 and not cluster.shutdown_time and not self.pending_tasks:
                cluster.shutdown(simulator.current_time)
                print(f"[Time {simulator.current_time}] Shutting down empty cluster {cluster.cluster_id}")
