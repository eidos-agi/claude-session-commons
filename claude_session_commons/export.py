"""Context export — generate markdown briefings from session data."""

from .display import relative_time


def export_context_md(session: dict, summary: dict | None,
                      deep: dict | None = None) -> str:
    """Generate a markdown context briefing for a session."""
    display = deep or summary or {}
    title = display.get("title", "Unknown session")
    project = session["project_dir"]
    age = relative_time(session["mtime"])
    duration = display.get("duration_fmt", "")
    stats_line = f", {duration} session" if duration else ""

    lines = [
        f"# Session Context: {title}",
        f"**Project:** {project}",
        f"**Last active:** {age}{stats_line}",
        "",
    ]

    goal = display.get("objective") or display.get("goal", "")
    if goal:
        lines += ["## What was being done", goal, ""]

    progress = display.get("progress") or display.get("what_was_done", "")
    if progress:
        lines += ["## Progress", progress, ""]

    state = display.get("state", "")
    if state:
        lines += ["## Where it left off", state, ""]

    next_steps = display.get("next_steps", "")
    if next_steps:
        lines += ["## Next steps", next_steps, ""]

    files = display.get("files", [])
    if files:
        lines.append("## Key files")
        for f in files[:8]:
            lines.append(f"- `{f}`")
        lines.append("")

    decisions = display.get("decisions_made", [])
    if decisions:
        lines.append("## Decisions made")
        for d in decisions:
            lines.append(f"- {d}")
        lines.append("")

    return "\n".join(lines)
