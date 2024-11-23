"""Create dataset for Flyte Llama fine-tuning.

This dataset should contain documents from the Flyte repositories for language
model fine-tuning.
"""

import itertools
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from typing import Iterable, Optional

from git import Repo

DEFAULT_EXTENSIONS = [
    ".py",
    ".md",
    ".rst",
    ".go",
    ".yaml",
    ".yml",
    ".json",
    ".js",
    ".tsx",
    ".ts",
    ".sh",
    ".txt",
    ".proto",
]
DEFAULT_INCLUDE_FILES = [
    "Dockerfile",
]
ROOT_URL = "https://github.com/"
REPO_URLS = [
    f"{ROOT_URL}cblmemo/skycamp24-tutorial",
    f"{ROOT_URL}skypilot-org/skypilot",
    f"{ROOT_URL}skypilot-org/skypilot-tutorial",
]


def iter_github_documents(
    url: str,
    repo_cache_dir: str,
    extensions: Optional[list[str]] = None,
    include_files: Optional[list[str]] = None,
    exclude_files: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> Iterable[str]:
    """Fetch documents from a github url."""
    extensions = extensions or DEFAULT_EXTENSIONS
    include_files = frozenset(include_files or DEFAULT_INCLUDE_FILES)
    exclude_files = frozenset(exclude_files or [])
    exclude_patterns = exclude_patterns or []
    repo_name = url.split("/")[-1]

    repo_dir = f'{repo_cache_dir}/{repo_name}'
    if os.path.exists(repo_dir):
        print(f"repo cache exists, loading from {repo_dir}")
        repo = Repo(repo_dir)
    else:
        repo = Repo.clone_from(url, repo_dir)

    git_sha = repo.head.commit.hexsha
    git_dir = Path(repo_cache_dir)

    exclude_from_patterns = frozenset(
        [*itertools.chain(*(git_dir.glob(p) for p in exclude_patterns))])

    for file in itertools.chain(
            *[git_dir.glob(f"{repo_name}/**/*{ext}") for ext in extensions]):
        if os.path.getsize(file) == 0:
            continue
        if (file.name not in include_files and
            (file.name in exclude_files or file in exclude_from_patterns)):
            continue

        github_url = f"{url}/blob/{git_sha}/{file.relative_to(git_dir)}"
        repo_filepath = file.relative_to(git_dir)
        yield file, repo_name, repo_filepath, github_url


def get_file_name(repo_filepath: Path) -> str:
    return "-".join(
        x.replace("/", "-")
        for x in str(repo_filepath).replace(ROOT_URL, "").split("/"))


def create_dataset(
    urls: list[str],
    output_dir: Path,
    repo_cache_dir: Path,
    **kwargs,
):
    """Create dataset with local temporary directory first."""
    import tempfile

    # !Edit for SkyPilot: Here we use a temporary directory to store the dataset.
    # Since mounted gcp bucket seems not so stable, we use this method to avoid
    # failed syscalls.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(os.path.join(temp_dir, 'output'))
        temp_cache = Path(os.path.join(temp_dir, 'cache'))

        print(
            f"Using temporary directories: \noutput: {temp_output}\ncache: {temp_cache}"
        )

        for url in urls:
            print(f"Processing url: {url}")
            for file, repo_name, repo_filepath, github_url in iter_github_documents(
                    url,
                    temp_cache,
                    **kwargs,
            ):
                file_name = get_file_name(repo_filepath)
                out_path = os.path.join(temp_output, repo_name, file_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                metadata_dir = os.path.join(temp_output, "metadata", repo_name)
                os.makedirs(metadata_dir, exist_ok=True)
                metadata_file = os.path.join(metadata_dir,
                                             f"{file_name}.metadata.json")

                print(f"Writing file: {out_path}")
                shutil.copy(file, out_path)

                metadata = {
                    "github_url": github_url,
                }
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f)

        print(
            f"Processing complete, copying to final destination: {output_dir}")

        # Create an archive
        archive_path = f"{temp_output}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(temp_output, arcname=".")
        print(f"Archive created at: {archive_path}")

        # Move the archive
        shutil.move(archive_path, output_dir)
        print('Archive moved to output directory')

        # Extract in the destination (if needed)
        compressed_file = f"{output_dir}/{os.path.basename(archive_path)}"
        os.system(f"tar -xzf {compressed_file} -C {output_dir}")
        if os.path.exists(compressed_file):
            os.remove(compressed_file)
            print(f"Deleted compressed file: {compressed_file}")
        else:
            print(f"File not found for deletion: {compressed_file}")

        print(f"Dataset created successfully at: {output_dir}")
