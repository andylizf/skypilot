import csv
from .algorithms.greedy_scheduler import GreedyScheduler
from .scheduler import Scheduler
from .cluster import Cluster, ClusterConfig, Task

class DynamicSchedulerSimulator:
    """Simulates dynamic scheduling with customizable algorithms."""
    def __init__(self, tasks: list[Task], cluster_config: ClusterConfig, scheduler_algorithm: Scheduler):
        self.tasks = sorted(tasks, key=lambda x: x.arrival_time)
        self.cluster_config = cluster_config
        self.clusters: list[Cluster] = []
        self.scheduler_algorithm = scheduler_algorithm
        self.current_time = 0
        self.finished_tasks: list[Task] = []
        self.total_cost = 0
        self.next_cluster_id = 1

    def create_cluster(self) -> Cluster:
        """Create a new cluster instance."""
        new_cluster = Cluster(
            cluster_id=self.next_cluster_id,
            config=self.cluster_config,
            start_time=self.current_time
        )
        self.next_cluster_id += 1
        self.clusters.append(new_cluster)
        return new_cluster

    def run_simulation(self):
        """Run the simulation loop."""
        while self.tasks or any(len(c.active_tasks) > 0 for c in self.clusters):
            new_tasks = self._process_new_tasks()
            self.scheduler_algorithm.schedule_tasks(self, new_tasks)
            self._update_time()
            self._process_finished_tasks()

        self._finalize_costs()

    def _update_time(self):
        """Advance the simulation time."""
        self.current_time += 1

    def _process_new_tasks(self)-> list[Task]:
        """Move newly arrived tasks to the pending queue."""
        new_tasks: list[Task] = []
        while self.tasks and self.tasks[0].arrival_time <= self.current_time:
            task = self.tasks.pop(0)
            new_tasks.append(task)
            print(f"[Time {self.current_time}] New task {task.task_id} arrived")
        return new_tasks

    def _process_finished_tasks(self):
        """Remove completed tasks from clusters."""
        for cluster in self.clusters:
            if not cluster.ready_time or cluster.shutdown_time:
                continue
            
            for task in list(cluster.active_tasks):
                if task.end_time is not None and task.end_time <= self.current_time:
                    cluster.active_tasks.remove(task)
                    self.finished_tasks.append(task)
                    print(f"[Time {self.current_time}] Task {task.task_id} completed on cluster {cluster.cluster_id}")

    def _finalize_costs(self):
        """Calculate the total cost of running clusters."""
        for cluster in self.clusters:
            if cluster.start_time:
                end_time = cluster.shutdown_time or self.current_time
                runtime = end_time - cluster.start_time
                self.total_cost += runtime * cluster.config.cost_per_minute

    def print_gantt(self):
        """Print a Gantt chart of the simulation."""
        CELL_WIDTH = 3
        
        print("\nGantt Chart:")
        print("Time:", end="")
        for t in range(self.current_time + 1):
            print(f"{t:^{CELL_WIDTH}}", end="")
        print()
        
        for cluster in sorted(self.clusters, key=lambda c: c.cluster_id):
            print(f"C{cluster.cluster_id:<2}: ", end="")
            for t in range(self.current_time + 1):
                if cluster.start_time is None or t < cluster.start_time:
                    symbol = " "
                elif cluster.shutdown_time is not None and t >= cluster.shutdown_time:
                    symbol = " "
                elif t < cluster.ready_time:
                    symbol = "⋯"
                else:
                    running_tasks = sum(
                        1 for task in self.finished_tasks + [task for cluster in self.clusters for task in cluster.active_tasks]
                        if (task.start_time is not None and 
                            task.end_time is not None and
                            task.start_time <= t < task.end_time and
                            any(cluster.cluster_id == c.cluster_id 
                                for c in self.clusters 
                                if task in c.active_tasks or task in self.finished_tasks and 
                                c.cluster_id == cluster.cluster_id))
                    )
                    
                    if running_tasks > 0:
                        symbol = "■"
                    else:
                        symbol = "□"
                
                print(f"{symbol:^{CELL_WIDTH}}", end="")
            print()

        print("\nLegend:")
        print("  ⋯ : Starting up")
        print("  ■ : Running tasks")
        print("  □ : Idle")
        print("    : Not active")

    def print_detailed_gantt(self):
        """Print a detailed Gantt chart showing task assignments."""
        CELL_WIDTH = 10
        
        print("\nDetailed Gantt Chart:")
        print("Time:", end="")
        for t in range(self.current_time + 1):
            print(f"{t:^{CELL_WIDTH}}", end="")
        print()
        
        print("     " + "─" * ((self.current_time + 1) * CELL_WIDTH))
        
        for cluster in sorted(self.clusters, key=lambda c: c.cluster_id):
            print(f"C{cluster.cluster_id:<2} │", end=" ")
            
            cluster_tasks = [task for task in self.finished_tasks + [t for c in self.clusters for t in c.active_tasks]
                            if any(c.cluster_id == cluster.cluster_id 
                                 for c in self.clusters 
                                 if task in c.active_tasks or task in self.finished_tasks)]
            
            for t in range(self.current_time + 1):
                if t < cluster.start_time:
                    symbol = " "
                elif cluster.shutdown_time is not None and t >= cluster.shutdown_time:
                    symbol = " "
                elif t < cluster.ready_time:
                    symbol = "⋯"
                else:
                    running_tasks = [
                        task for task in cluster_tasks
                        if task.start_time is not None and 
                           task.end_time is not None and
                           task.start_time <= t < task.end_time
                    ]
                    if running_tasks:
                        task_ids = [str(task.task_id) for task in running_tasks]
                        if len(task_ids) > 1:
                            symbol = f"{','.join(task_ids)}"
                        else:
                            symbol = f"T{task_ids[0]}"
                    else:
                        symbol = "·"
                
                print(f"{symbol:^{CELL_WIDTH}}", end="")
            print()
            
            print("    │", end=" ")
            for t in range(self.current_time + 1):
                if t < cluster.start_time:
                    gpu_usage = " "
                elif cluster.shutdown_time is not None and t >= cluster.shutdown_time:
                    gpu_usage = " "
                elif t < cluster.ready_time:
                    gpu_usage = "⋯"
                else:
                    running_tasks = [
                        task for task in cluster_tasks
                        if task.start_time is not None and 
                           task.end_time is not None and
                           task.start_time <= t < task.end_time
                    ]
                    if running_tasks:
                        gpu_count = sum(task.gpu_requirement for task in running_tasks)
                        total_gpus = cluster.config.gpu_count
                        gpu_usage = f"{gpu_count}/{total_gpus}"
                    else:
                        gpu_usage = "0/4"
                print(f"{gpu_usage:^{CELL_WIDTH}}", end="")
            print(" (GPUs)")
            
            print("    " + "├" + "─" * ((self.current_time + 1) * CELL_WIDTH))
        
        print("\nLegend:")
        print("  ⋯     : Starting up")
        print("  T1    : Single task (Task ID 1)")
        print("  1,2,3 : Multiple tasks (Task IDs)")
        print("  ·     : Idle")
        print("  n/m   : n GPUs in use out of m total")
        print("        : Not active")

    def print_summary(self):
        """Print a summary of the simulation."""
        self.print_gantt()
        self.print_detailed_gantt()
        print(f"\nSimulation finished at time {self.current_time}")
        print(f"Total tasks completed: {len(self.finished_tasks)}")
        print(f"Total cost: ${self.total_cost:.2f}")


def load_tasks(task_file: str) -> list[Task]:
    """Load tasks from a CSV file."""
    import os
    file_path = os.path.join(os.path.dirname(__file__), task_file)
    tasks: list[Task] = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(Task(
                task_id=int(row['task_id']),
                arrival_time=int(row['arrival_time']),
            ))
    return tasks


def main():
    # Load tasks
    tasks = load_tasks("tasks.csv")
    
    # Create cluster configuration
    cluster_config = ClusterConfig(
        gpu_count=4,
        startup_time=2,
        cost_per_minute=0.4
    )

    # Initialize the scheduler with the configuration
    scheduler_algorithm = GreedyScheduler()

    # Run the simulation
    simulator = DynamicSchedulerSimulator(tasks, cluster_config, scheduler_algorithm)
    simulator.run_simulation()
    simulator.print_summary()


if __name__ == "__main__":
    main()
