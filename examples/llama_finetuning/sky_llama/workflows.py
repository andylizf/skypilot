import os
from typing import List, Optional
import sys
import logging

import sky_llama

logger = logging.getLogger("sky_llama")

def batch_size_tuning_configs(
    batch_sizes: List[int],
) -> List[sky_llama.train.TrainerConfig]:
    configs = []
    for batch_size in batch_sizes:
        configs.append(replace(config, batch_size=batch_size))
    return configs

def train() -> None:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Training Flyte Llama")

    wandb_run_name = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID", "local")
    os.environ["WANDB_RUN_ID"] = wandb_run_name

    hf_auth_token = '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'

    sky_llama.train.train(hf_auth_token)

def create_dataset(additional_urls: Optional[List[str]] = None) -> None:
    urls = [*sky_llama.dataset.REPO_URLS, *(additional_urls or [])]

    output_dir = '/mnt/data/dataset'
    repo_cache_dir = '/mnt/data/repo_cache'

    sky_llama.dataset.create_dataset(urls, output_dir, repo_cache_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Entry function not specified. Please provide a function name.")
    
    function_name = sys.argv[1]
    
    try:
        # Dynamically call the function based on its name
        globals()[function_name]()
    except KeyError:
        raise ValueError(f"Function '{function_name}' not found.")