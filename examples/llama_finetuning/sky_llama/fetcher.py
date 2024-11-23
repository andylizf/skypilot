"""Utility functions for issue processing and LLM tasks."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline


@dataclass
class Issue:
    title: str
    body: str
    number: int
    state: str
    comments: List[str]
    url: str


@dataclass
class Discussion:
    title: str
    body: str
    number: int
    comments: List[str]
    url: str


def fetch_github_issues(owner: str, repo: str, token: str) -> List[Issue]:
    """Fetch all issues from a GitHub repository."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    issues = []
    page = 1
    while True:
        # Fetch both open and closed issues
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            "state": "all",
            "page": page,
            "per_page": 100,
            "sort": "created",
            "direction": "desc"
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            break

        page_issues = response.json()
        if not page_issues:
            break

        for issue in page_issues:
            # Skip pull requests
            if "pull_request" in issue:
                continue

            # Fetch comments
            comments_url = issue["comments_url"]
            comments_response = requests.get(comments_url, headers=headers)
            comments = [c["body"] for c in comments_response.json()
                       ] if comments_response.status_code == 200 else []

            issues.append(
                Issue(title=issue["title"],
                      body=issue["body"] or "",
                      number=issue["number"],
                      state=issue["state"],
                      comments=comments,
                      url=issue["html_url"]))

        page += 1

    return issues


def fetch_github_discussions(owner: str, repo: str,
                             token: str) -> List[Dict[str, Any]]:
    """Fetch all discussions from a GitHub repository using GraphQL API."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    query = """
    query($owner: String!, $repo: String!, $cursor: String) {
      repository(owner: $owner, name: $repo) {
        discussions(first: 100, after: $cursor) {
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            title
            body
            number
            url
            comments(first: 100) {
              nodes {
                body
              }
            }
          }
        }
      }
    }
    """

    discussions = []
    cursor = None

    while True:
        variables = {"owner": owner, "repo": repo, "cursor": cursor}

        response = requests.post("https://api.github.com/graphql",
                                 headers=headers,
                                 json={
                                     "query": query,
                                     "variables": variables
                                 })

        if response.status_code != 200:
            break

        data = response.json()
        discussions_data = data["data"]["repository"]["discussions"]

        for discussion in discussions_data["nodes"]:
            comments = [c["body"] for c in discussion["comments"]["nodes"]]
            discussions.append({
                "title": discussion["title"],
                "body": discussion["body"],
                "number": discussion["number"],
                "url": discussion["url"],
                "comments": comments
            })

        if not discussions_data["pageInfo"]["hasNextPage"]:
            break

        cursor = discussions_data["pageInfo"]["endCursor"]

    return discussions


def filter_issues_with_llm(
    issues: List[Issue],
    discussions: List[Dict[str, Any]],
    model_path: str,
    token: Optional[str] = None,
) -> Tuple[List[Union[Issue, Discussion]], List[Dict]]:
    """Filter issues and discussions using Llama as classifier."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        token=token,
        trust_remote_code=True,
        load_in_4bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              token=token,
                                              trust_remote_code=True)

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto")

    SPLITTER = "Answer with exactly one word (YES/NO):"

    PROMPT_TEMPLATE = """You are an expert at analyzing GitHub issues and discussions. Determine if the following content is a feature request that would require code changes and implementation work.

Content Type: {content_type}
Title: {title}
Description: {description}

A valid feature request should meet ALL these criteria:
1. Requests new or enhanced functionality
2. Requires changes to the codebase
3. Is not a bug report or general question
4. Has clear technical implementation needs

Examples:
- "Add support for AWS Graviton instances" (YES - new functionality)
- "Documentation is unclear" (NO - no code changes needed)
- "Program crashes when using spot instances" (NO - bug report)
- "How to configure auto-scaling?" (NO - usage question)

Based on these criteria, is this a feature request requiring implementation? """ + SPLITTER

    filtered_items = []
    classification_results = []

    # Process issues
    for item in issues:
        try:
            prompt = PROMPT_TEMPLATE.format(content_type="Issue",
                                            title=item.title,
                                            description=item.body[:1000])

            result = pipe(prompt,
                          max_new_tokens=20,
                          temperature=0.1,
                          do_sample=False)

            response = result[0]['generated_text'].split(SPLITTER)[-1].strip()

            if "YES" in response.upper():
                label = 1
            else:
                label = 0

            result_info = {
                "issue_number": item.number,
                "title": item.title,
                "classification": {
                    "label": label,
                    "category": "FEATURE_REQUEST" if label == 1 else "OTHER",
                    "raw_response": result[0]['generated_text']
                },
                "accepted": label == 1,
                "source": "issue"
            }

            if label == 1:
                filtered_items.append(item)

            classification_results.append(result_info)

        except Exception as e:
            print(f"Error processing issue #{item.number}: {e}")
            classification_results.append({
                "issue_number": item.number,
                "title": item.title,
                "error": str(e),
                "accepted": False,
                "source": "issue"
            })

    # Process discussions
    for disc in discussions:
        try:
            prompt = PROMPT_TEMPLATE.format(content_type="Discussion",
                                            title=disc["title"],
                                            description=disc["body"][:1000])

            result = pipe(prompt,
                          max_new_tokens=20,
                          temperature=0.1,
                          do_sample=False)

            response = result[0]['generated_text'].split(" ")[-1].strip()

            if "YES" in response.upper():
                label = 1
            else:
                label = 0

            result_info = {
                "discussion_number": disc["number"],
                "title": disc["title"],
                "classification": {
                    "label": label,
                    "category": "FEATURE_REQUEST" if label == 1 else "OTHER",
                    "raw_response": result[0]['generated_text']
                },
                "accepted": label == 1,
                "source": "discussion"
            }

            if label == 1:
                discussion_obj = Discussion(title=disc["title"],
                                            body=disc["body"],
                                            number=disc["number"],
                                            comments=disc["comments"],
                                            url=disc["url"])
                filtered_items.append(discussion_obj)

            classification_results.append(result_info)

        except Exception as e:
            print(f"Error processing discussion #{disc['number']}: {e}")
            classification_results.append({
                "discussion_number": disc["number"],
                "title": disc["title"],
                "error": str(e),
                "accepted": False,
                "source": "discussion"
            })

    return filtered_items, {
        "results": classification_results,
        "statistics": {
            "total": len(issues) + len(discussions),
            "total_issues": len(issues),
            "total_discussions": len(discussions),
            "feature_requests_from_issues": sum(
                1 for r in classification_results
                if r.get("source") == "issue" and r.get("accepted")),
            "feature_requests_from_discussions": sum(
                1 for r in classification_results
                if r.get("source") == "discussion" and r.get("accepted")),
        }
    }


def save_to_dataset(issues: List[Issue], discussions: List[Dict[str, Any]],
                    output_dir: Path):
    """Save fetched issues and discussions to dataset directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    issues_dir = output_dir / "issues"
    issues_dir.mkdir(exist_ok=True)

    for issue in issues:
        issue_file = issues_dir / f"issue_{issue.number}.json"
        with issue_file.open("w") as f:
            json.dump(issue.__dict__, f, indent=2)

    discussions_dir = output_dir / "discussions"
    discussions_dir.mkdir(exist_ok=True)

    for discussion in discussions:
        discussion_file = discussions_dir / f"discussion_{discussion['number']}.json"
        with discussion_file.open("w") as f:
            json.dump(discussion, f, indent=2)
