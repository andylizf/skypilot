import os
import subprocess
import time
import re
from collections import defaultdict

# Dictionary to store job data
job_data = defaultdict(lambda: {
    "clusters": defaultdict(lambda: {"duration": 0, "starting_duration": 0}),
    "current_region": None,
})


def parse_jobs_queue_output(output, elapsed_time):
    """Parse the 'sky jobs queue --all' command output."""
    lines = output.split("\n")
    updated_jobs = {}

    task_line_indicator = "↳"

    for line in lines:
        if line.strip() == "" or line.startswith("Fetching") or line.startswith("Managed jobs") or "In progress tasks" in line:
            continue

        if task_line_indicator in line:
            # Use regular expression to handle multiple spaces
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) < 13:  # Minimum expected fields for a valid task line
                continue

            # Extract Task ID
            task_id = parts[1].replace(task_line_indicator, "").strip()

            # Extract STATUS
            status = parts[8]  # "SUCCEEDED", "RUNNING", "STARTING", etc.

            # Extract CLUSTER and REGION
            cluster = parts[10] if parts[10] != "-" else None
            region = parts[11] if parts[11] != "-" else None

            if (status in ["RUNNING", "STARTING"]) and cluster and region:
                key = f"{cluster} {region}"

                # Initialize job data if not present
                if task_id not in job_data:
                    job_data[task_id] = {
                        "clusters": defaultdict(lambda: {"duration": 0.0, "starting_duration": 0.0}),
                        "current_region": key,
                    }

                # Update durations for the specific key
                clusters = job_data[task_id]["clusters"]
                if job_data[task_id]["current_region"] != key:
                    # Transition to a new cluster/region
                    job_data[task_id]["current_region"] = key
                    if key not in clusters:
                        clusters[key] = {"duration": 0.0, "starting_duration": 0.0}
                
                if status == "RUNNING":
                    clusters[key]["duration"] += elapsed_time
                elif status == "STARTING":
                    clusters[key]["starting_duration"] += elapsed_time

                # Add to updated_jobs for printing
                updated_jobs[task_id] = job_data[task_id]
    
    return updated_jobs

def poll_jobs():
    """Poll job statuses at regular intervals."""
    try:
        last_poll_time = time.time()
        while True:
            # Change directory and activate environment
            os.chdir(os.path.expanduser("~/Desktop/projects/skypilot"))

            # Run the SkyPilot jobs command
            result = subprocess.run(["sky", "jobs", "queue", "--all"], capture_output=True, text=True)
            if result.returncode != 0:
                print("Failed to fetch job statuses.")
                continue

            output = result.stdout
            updated_jobs = parse_jobs_queue_output(output, time.time() - last_poll_time)
            last_poll_time = time.time()

            # Print updated jobs
            for job_id, data in updated_jobs.items():
                print(f"Job {job_id}:")
                for key, durations in data["clusters"].items():
                    print(f"  Cluster/Region: {key}")
                    print(f"    RUNNING Duration: {durations['duration']} mins")
                    print(f"    STARTING Duration: {durations['starting_duration']} mins")

            time.sleep(30)  # Poll every 5 minutes
    except KeyboardInterrupt:
        # Print job data on termination
        print("\nFinal Job Data:")
        for job_id, data in job_data.items():
            print(f"Job {job_id}:")
            for key, durations in data["clusters"].items():
                print(f"  Cluster/Region: {key}")
                print(f"    RUNNING Duration: {durations['duration']} mins")
                print(f"    STARTING Duration: {durations['starting_duration']} mins")
        print("Exiting polling loop.")

if __name__ == "__main__":
    poll_jobs()