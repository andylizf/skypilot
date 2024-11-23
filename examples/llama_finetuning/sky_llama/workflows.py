from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

import sky_llama
import transformers

logger = logging.getLogger("sky_llama")


def batch_size_tuning_configs(
    batch_sizes: List[int],) -> List[sky_llama.train.TrainerConfig]:
    configs = []
    for batch_size in batch_sizes:
        configs.append(replace(config, batch_size=batch_size))
    return configs


def train() -> None:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Training Sky Llama")

    wandb_run_name = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID", "local")
    os.environ["WANDB_RUN_ID"] = wandb_run_name
    os.environ["WANDB_API_KEY"] = '5238e0a0a01e1ce6b9c8ddf2def51c9e3dade0da'

    hf_auth_token = 'xxxx'

    sky_llama.train.train(hf_auth_token)


def create_dataset(additional_urls: Optional[List[str]] = None) -> None:
    urls = [*sky_llama.dataset.REPO_URLS, *(additional_urls or [])]

    output_dir = 'sky_llama/data/dataset'
    repo_cache_dir = 'sky_llama/data/repo_cache'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(repo_cache_dir, exist_ok=True)

    sky_llama.dataset.create_dataset(urls, output_dir, repo_cache_dir)


def fetch_github_data() -> None:
    """Fetch GitHub issues and discussions."""
    github_token = 'xxxxx'
    owner = 'skypilot-org'
    repo = 'skypilot'
    output_dir = 'sky_llama/data'

    output_path = Path(output_dir)
    issues = sky_llama.fetcher.fetch_github_issues(owner, repo, github_token)
    discussions = sky_llama.fetcher.fetch_github_discussions(
        owner, repo, github_token)

    sky_llama.fetcher.save_to_dataset(issues, discussions, output_path)


def filter_issues() -> None:
    """Filter issues using LLM to identify real problems."""
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    hf_token = 'xxxx'

    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    filtered_dir = Path("sky_llama/filtered_issues")
    filtered_dir.mkdir(parents=True, exist_ok=True)

    working_dir = Path.cwd()
    print(f"Working directory: {working_dir}")
    print(f"Filtered directory: {filtered_dir}")

    # Load issues
    issues_dir = Path('sky_llama/input/issues')
    print(f"Issues directory: {issues_dir}")
    issue_files = list(issues_dir.glob("*.json"))
    print(f"Issues directory contents: {issue_files}")

    issues: List[sky_llama.fetcher.Issue] = []
    for issue_file in issue_files:
        print(f"Loading issue file: {issue_file}")
        try:
            with issue_file.open() as f:
                data = json.load(f)
                issues.append(sky_llama.fetcher.Issue(**data))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error loading issue file {issue_file}: {e}")

    print(f"Loaded {len(issues)} issues")

    # Load discussions
    discussions_dir = Path('sky_llama/input/discussions')
    discussion_files = list(discussions_dir.glob("*.json"))
    print(f"Discussions directory contents: {discussion_files}")

    discussions: List[Dict] = []
    for disc_file in discussion_files:
        try:
            with disc_file.open() as f:
                discussions.append(json.load(f))
        except json.JSONDecodeError as e:
            print(f"Error loading discussion file {disc_file}: {e}")

    print(f"Loaded {len(discussions)} discussions")

    # Filter issues using LLM
    try:
        filtered_items, classification_results = sky_llama.fetcher.filter_issues_with_llm(
            issues, discussions, model_path, token=hf_token)
    except Exception as e:
        print(f"Error filtering issues with LLM: {e}")
        return

    print(f"Filtered {len(filtered_items)} items")

    # Save classification results
    classification_results_path = filtered_dir / "classification_results.json"
    with classification_results_path.open("w") as f:
        json.dump(
            {
                "total_items": len(issues) + len(discussions),
                "filtered_items": len(filtered_items),
                "results": classification_results,
                "timestamp": datetime.now().isoformat()
            },
            f,
            indent=2)

    # Save filtered items
    saved_files = []
    for item in filtered_items:
        if isinstance(item, sky_llama.fetcher.Issue):
            output_file = filtered_dir / f"filtered_issue_{item.number}.json"
        else:  # Assume it's a discussion
            output_file = filtered_dir / f"filtered_discussion_{item['id']}.json"

        try:
            with output_file.open("w") as f:
                json.dump(item.__dict__ if hasattr(item, '__dict__') else item,
                          f,
                          indent=2)
            saved_files.append(output_file)
        except Exception as e:
            print(f"Error saving item {item}: {e}")

    # Save manifest
    manifest_path = filtered_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(
            {
                "total_issues": len(issues),
                "filtered_items": len(filtered_items),
                "files": [str(f.name) for f in saved_files],
                "timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "classification_summary": {
                    "accepted": len(filtered_items),
                    "rejected": len(issues) - len(filtered_items)
                }
            },
            f,
            indent=2)

    # Log final state
    final_contents = list(filtered_dir.glob("*"))
    total_size = sum(f.stat().st_size for f in final_contents)
    print(f"Final directory contents: {final_contents}")
    print(f"Final directory size: {total_size} bytes")


def generate_answers() -> None:
    print(f"Transformers version: {transformers.__version__}")
    hf_token = 'xxxx'
    model_path = "meta-llama/Llama-3.1-8B-Instruct"

    issues_path = Path('sky_llama/input')
    print(f"Reading filtered issues from: {issues_path}")
    print(f"Directory contents: {list(issues_path.glob('*'))}")

    # 加载issues
    issues = []
    for issue_file in issues_path.glob("filtered_*.json"):
        print(f"Loading issue file: {issue_file}")
        with issue_file.open() as f:
            try:
                issue_data = json.load(f)
                issues.append(issue_data)
            except Exception as e:
                print(f"Error loading {issue_file}: {e}")

    print(f"Loaded {len(issues)} issues")

    # 设置环境变量
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    adapter_path = Path('sky_llama/model')

    # 配置模型
    serving_config = sky_llama.inference.ServingConfig(
        model_path=model_path,
        adapter_path=str(adapter_path),  # Convert Path to string
        # max_new_tokens=8192,
        device_map="auto",
        hf_token=hf_token,
    )

    # 加载模型和生成答案
    model, tokenizer = sky_llama.inference.load_pipeline(serving_config)
    answers = []

    system_prompt = """You are an expert programmer analyzing a feature request for the SkyPilot framework. Break down this feature request into a clear implementation plan.

IMPORTANT:
1. Do not include commit hashes, version numbers, or any placeholder text like "Commit hash".
2. Be concise and to the point, do not include any other information.
3. Use markdown syntax for code blocks and lists.
4. Do not write like an email, just write the plan.

Here's an example analysis of a fake feature request:
----------------
Feature Request:
Title: Add timeout parameter for spot job recovery
Description: When using spot instances, if the instance is preempted, SkyPilot will try to recover indefinitely. We should add a timeout parameter to limit the recovery time.

Analysis:
1. Current Behavior:
- Spot recovery loop runs without timeout control:
```python
# spot.py
def recover_spot_job(task: Task):
    while True:  # Infinite loop without timeout
        try:
            return _attempt_spot_recovery(task)
        except Exception as e:
            logger.error(f'Recovery failed: {e}')
            time.sleep(RETRY_INTERVAL)
```
- Task class has no timeout configuration:
```python
class Task:
    def __init__(self, spot_policy: Optional[str] = None):
        self.spot_policy = spot_policy  # No timeout parameter
```
- Users can only interrupt recovery manually by killing the process

2. Proposed Solution:
- Add spot_recovery_timeout parameter to job submission
- Implement timeout logic in spot recovery loop
- Provide clear error message when timeout is reached

3. Implementation Plan:
- Modify spot.py to add timeout parameter and logic
- Update task.py to include timeout configuration
- Add timeout handling in recovery loop

4. Code Implementation:
- Add timeout parameter to Task class:
```python
class Task:
    def __init__(self, spot_recovery_timeout: Optional[int] = None):
        self.spot_recovery_timeout = spot_recovery_timeout
```

- Implement timeout logic in spot recovery:
```python
def recover_spot_job(task: Task):
    start_time = time.time()
    while True:
        if (task.spot_recovery_timeout and 
            time.time() - start_time > task.spot_recovery_timeout):
            raise SpotTimeoutError(
                f'Spot recovery timeout after {task.spot_recovery_timeout}s')
        try:
            return _attempt_spot_recovery(task)
        except Exception as e:
            logger.error(f'Recovery failed: {e}')
```

- Add unit tests:
```python
def test_spot_recovery_timeout():
    task = Task(spot_recovery_timeout=60)
    with pytest.raises(SpotTimeoutError):
        recover_spot_job(task)
```

----------------

Provide a similar detailed analysis with concrete code examples.

1. Current Behavior:
- Describe current implementation with relevant code snippets in SkyPilot repository
- Identify limitations in existing code
- Show where changes will be needed

2. Proposed Solution:
- Core functionality needed
- Key benefits
- Technical requirements

3. Implementation Plan:
- Components to modify
- Key steps (max 3)
- Integration points

4. Code Implementation:
- Key function/class changes with code examples
- Integration code
- Testing approach with example test cases

Keep each section brief and focused. Prioritize practical implementation details."""

    # Modify the prompt template
    for issue in issues[:100]:
        try:
            issue_prompt = f"Title: {issue['title']}\nDescription: {issue['body']}\n"
            full_prompt = system_prompt + "\n\n" + issue_prompt

            print(
                f"Generating implementation plan for feature #{issue['number']}"
            )
            print(f"Issue prompt: {issue_prompt}")
            print(f"Input prompt length: {len(full_prompt)}")

            inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs,
                                     temperature=0.2,
                                     top_p=0.75,
                                     top_k=40,
                                     num_beams=1,
                                     no_repeat_ngram_size=3,
                                     repetition_penalty=1.2,
                                     encoder_repetition_penalty=1.0,
                                     typical_p=1.0,
                                     length_penalty=1.2,
                                     do_sample=True,
                                     max_new_tokens=1024,
                                     pad_token_id=tokenizer.eos_token_id,
                                     eos_token_id=tokenizer.eos_token_id)

            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True)
            response = response.strip()
            print("Final answer:", response)

            # Add metadata to track generation parameters
            answers.append({
                "issue_number": issue["number"],
                "issue_url": issue["url"],
                "answer": response,
                "generation_params": {
                    "temperature": 0.2,
                    "max_new_tokens": 2048,
                    "repetition_penalty": 1.2,
                    "top_p": 0.75,
                    "top_k": 40,
                    "num_beams": 1,
                    "no_repeat_ngram_size": 3,
                    "encoder_repetition_penalty": 1.0,
                    "typical_p": 1.0,
                    "length_penalty": 1.2
                }
            })

        except Exception as e:
            print(
                f"Error generating answer for issue #{issue.get('number', 'unknown')}: {e}"
            )
            continue

    # 保存答案
    output_path = Path("/tmp/issue_answers")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(answers)} answers to {output_path}")
    with (output_path / "answers.json").open("w") as f:
        json.dump(
            {
                "total_issues": len(issues),
                "total_answers": len(answers),
                "answers": answers,
                "timestamp": datetime.now().isoformat()
            },
            f,
            indent=2)

    print(f"Answers saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Entry function not specified. Please provide a function name.")

    function_name = sys.argv[1]

    try:
        # Dynamically call the function based on its name
        globals()[function_name]()
    except KeyError:
        raise ValueError(f"Function '{function_name}' not found.")
