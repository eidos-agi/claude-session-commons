"""Git context extraction for session projects."""

import os
import subprocess


def get_git_context(project_dir: str) -> dict:
    """Pull recent commits and uncommitted changes from a project's git repo.

    Returns dict with:
        recent_commits: str — last 5 commits (oneline + stat)
        uncommitted_changes: str — diff stat + untracked files
        is_git_repo: bool
    """
    result = {"recent_commits": "", "uncommitted_changes": "", "is_git_repo": False}

    if not os.path.isdir(project_dir):
        return result

    try:
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if check.returncode != 0:
            return result
        result["is_git_repo"] = True

        log = subprocess.run(
            ["git", "log", "--oneline", "--stat", "--no-color", "-5"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if log.returncode == 0:
            result["recent_commits"] = log.stdout.strip()[:2000]

        diff = subprocess.run(
            ["git", "diff", "--stat", "--no-color", "HEAD"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if diff.returncode == 0 and diff.stdout.strip():
            result["uncommitted_changes"] = diff.stdout.strip()[:1000]

        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if untracked.returncode == 0 and untracked.stdout.strip():
            files = untracked.stdout.strip().split("\n")[:10]
            if files:
                result["uncommitted_changes"] += "\n\nUntracked files:\n" + "\n".join(files)

    except (subprocess.TimeoutExpired, Exception):
        pass

    return result


def get_git_anchor(project_dir: str) -> dict:
    """Extract structured git anchor data for entity indexing.

    Returns dict with:
        branch: str — current branch name
        commit_hashes: list[str] — last 5 short hashes
        commit_messages: list[str] — last 5 oneline messages
        files_changed: list[str] — files from recent commits (deduplicated)
    """
    result = {
        "branch": "",
        "commit_hashes": [],
        "commit_messages": [],
        "files_changed": [],
    }

    if not os.path.isdir(project_dir):
        return result

    try:
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if check.returncode != 0:
            return result

        # Current branch
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if branch.returncode == 0:
            result["branch"] = branch.stdout.strip()

        # Last 5 commits (hash + message)
        log = subprocess.run(
            ["git", "log", "--oneline", "--no-color", "-5"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if log.returncode == 0 and log.stdout.strip():
            for line in log.stdout.strip().splitlines():
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    result["commit_hashes"].append(parts[0])
                    result["commit_messages"].append(parts[1])

        # Files changed in recent commits
        diff_files = subprocess.run(
            ["git", "log", "--name-only", "--pretty=format:", "-5"],
            capture_output=True, text=True, cwd=project_dir, timeout=5,
        )
        if diff_files.returncode == 0 and diff_files.stdout.strip():
            seen = set()
            for line in diff_files.stdout.strip().splitlines():
                f = line.strip()
                if f and f not in seen:
                    seen.add(f)
                    result["files_changed"].append(f)

    except (subprocess.TimeoutExpired, Exception):
        pass

    return result


def has_uncommitted_changes(project_dir: str) -> bool:
    """Quick check for uncommitted git changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=project_dir, timeout=3,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False
