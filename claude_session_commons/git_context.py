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
