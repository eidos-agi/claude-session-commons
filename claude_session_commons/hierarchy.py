"""Hierarchical memory — topic discovery and multi-level summarization.

Every project gets topic discovery, regardless of size. The LLM groups
session summaries into natural workstreams/topics and summarizes each.

Two paths to the same output:
- Small projects (< ~100 sessions): LLM does grouping + summarization in one call
- Large projects (≥ ~100 sessions): HDBSCAN pre-clusters embeddings, LLM summarizes each cluster

Output: multiple L2 topic summaries per project, stored in summary_levels.
"""

import json
import subprocess
from pathlib import Path

from .cache import SessionCache

# ── Schemas ──────────────────────────────────────────────────

TOPIC_DISCOVERY_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic_name": {"type": "string"},
                    "status": {"type": "string"},
                    "narrative": {"type": "string"},
                    "key_decisions": {"type": "array", "items": {"type": "string"}},
                    "open_threads": {"type": "array", "items": {"type": "string"}},
                    "session_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["topic_name", "status", "narrative", "session_ids"],
            },
        },
    },
    "required": ["topics"],
})

CLUSTER_SUMMARY_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "topic_name": {"type": "string"},
        "status": {"type": "string"},
        "narrative": {"type": "string"},
        "key_decisions": {"type": "array", "items": {"type": "string"}},
        "open_threads": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["topic_name", "status", "narrative"],
})

PORTFOLIO_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "active_projects": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "status": {"type": "string"},
                "one_liner": {"type": "string"},
            },
        }},
        "stalled_projects": {"type": "array", "items": {"type": "string"}},
        "themes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "active_projects", "stalled_projects", "themes"],
})

# Threshold: above this, use HDBSCAN pre-clustering
LLM_DIRECT_THRESHOLD = 100
# Token budget per call (~8k tokens ≈ ~32k chars)
MAX_INPUT_CHARS = 32000


# ── LLM ──────────────────────────────────────────────────────

def _call_claude(prompt: str, schema: str, timeout: int = 120) -> dict | None:
    """Call claude -p with structured JSON output. Returns None on failure."""
    try:
        cmd = [
            "claude", "-p", prompt,
            "--no-session-persistence", "--output-format", "json",
            "--model", "claude-haiku-4-5-20251001",
            "--json-schema", schema,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        output = result.stdout.strip()
        parsed = json.loads(output)
        if "structured_output" in parsed and isinstance(parsed["structured_output"], dict):
            return parsed["structured_output"]
        if "result" in parsed and isinstance(parsed["result"], dict):
            return parsed["result"]
        if "result" in parsed and isinstance(parsed["result"], str):
            return json.loads(parsed["result"])
        return parsed
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


# ── Gather L1 summaries ─────────────────────────────────────

def _gather_session_summaries(
    project_path: str,
    conn,
    cache: SessionCache | None = None,
) -> list[dict]:
    """Gather L1 session summaries for a project.

    Reads from the resume-summaries cache (which has rich summaries from
    summarize_quick: title, goal, what_was_done, state). Falls back to
    chunk content if cache miss.

    Returns list of {session_id, summary, timestamp}.
    """
    rows = conn.execute(
        """SELECT DISTINCT session_id, MAX(timestamp) as last_ts
           FROM chunks
           WHERE project_path = ?
           GROUP BY session_id
           ORDER BY last_ts DESC""",
        (project_path,),
    ).fetchall()

    # Build a secondary cache that reads directly from JSON files
    # The SessionCache needs a cache_key from file mtime, but we can
    # scan the cache directory for matching session IDs.
    cached_summaries = {}
    if cache is not None:
        _load_cached_summaries(cache, cached_summaries)

    summaries = []
    for session_id, last_ts in rows:
        summary_text = None

        # Try pre-loaded cache
        if session_id in cached_summaries:
            s = cached_summaries[session_id]
            title = s.get("title", "")
            goal = s.get("goal", "")
            what = s.get("what_was_done", "")
            state = s.get("state", "")
            summary_text = f"{title}. {goal} {what} {state}".strip()

        # Fall back to chunk content
        if not summary_text:
            chunk_row = conn.execute(
                """SELECT content FROM chunks
                   WHERE session_id = ? AND chunk_type = 'turn'
                   ORDER BY timestamp DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            if chunk_row:
                summary_text = chunk_row[0][:500]

        if summary_text:
            summaries.append({
                "session_id": session_id,
                "summary": summary_text,
                "timestamp": last_ts or "",
            })

    return summaries


def _load_cached_summaries(cache: SessionCache, out: dict):
    """Scan cache directory for session summaries, bypassing cache_key.

    The normal cache.get() requires a cache_key derived from the session
    file's mtime. We don't have the file path here, so we read the JSON
    files directly from the cache directory.
    """
    cache_dir = cache._dir
    if not cache_dir.exists():
        return

    for f in cache_dir.iterdir():
        if not f.suffix == ".json":
            continue
        try:
            data = json.loads(f.read_text())
            session_id = f.stem
            summary = data.get("summary")
            if summary and isinstance(summary, dict):
                out[session_id] = summary
        except (json.JSONDecodeError, OSError):
            continue


# ── Topic discovery (LLM-native, small projects) ────────────

def _discover_topics_llm(
    project_path: str,
    session_summaries: list[dict],
) -> list[dict] | None:
    """Use the LLM to group sessions into topics and summarize each.

    Single call: the model discovers natural workstreams/topics,
    assigns sessions, and generates per-topic summaries.
    """
    project_name = Path(project_path).name if project_path else "unknown"

    sessions_block = "\n\n".join(
        f"[{s['session_id'][:8]}] [{s['timestamp'][:10]}] {s['summary'][:500]}"
        for s in session_summaries
    )

    # Cap input
    if len(sessions_block) > MAX_INPUT_CHARS:
        sessions_block = sessions_block[:MAX_INPUT_CHARS]

    prompt = f"""You are analyzing {len(session_summaries)} Claude Code sessions from the project "{project_name}" ({project_path}).

These sessions may cover multiple distinct workstreams, initiatives, or topics. Your job is to discover the natural topic groupings and summarize each one.

SESSIONS (most recent first):
{sessions_block}

Group these sessions by topic/workstream. A session can only belong to one topic. Find the natural groupings — there might be 1 topic or 10, depending on the data. Don't force topics that aren't there, but don't merge distinct workstreams either.

For each topic, return:
- "topic_name": Short descriptive name (e.g., "Wrike Renewal & SCR-001", "Red Team Framework")
- "status": Current status in 1 sentence
- "narrative": What this workstream is about, what's been done, where it stands (3-5 sentences)
- "key_decisions": Important decisions made (up to 3)
- "open_threads": Unresolved work items (up to 3)
- "session_ids": Array of session ID prefixes (the 8-char IDs from the brackets) that belong to this topic

Return raw JSON only."""

    result = _call_claude(prompt, TOPIC_DISCOVERY_SCHEMA)
    if not result:
        return None

    return result.get("topics", [])


# ── Topic discovery (HDBSCAN, large projects) ───────────────

def _discover_topics_clustered(
    project_path: str,
    session_summaries: list[dict],
    conn,
) -> list[dict] | None:
    """Use embedding clustering + LLM for large projects.

    1. Get embeddings for each session's chunks from the DB
    2. HDBSCAN to find topic clusters
    3. LLM summarizes each cluster
    """
    try:
        import numpy as np
        from hdbscan import HDBSCAN
    except ImportError:
        # Fall back to LLM-native in batches if HDBSCAN not available
        return _discover_topics_llm_batched(project_path, session_summaries)

    # Get average embedding per session from existing chunk embeddings
    session_ids = [s["session_id"] for s in session_summaries]
    summary_by_id = {s["session_id"]: s for s in session_summaries}

    embeddings = []
    valid_session_ids = []

    for sid in session_ids:
        row = conn.execute(
            """SELECT AVG(embedding) FROM (
                SELECT v.embedding FROM vec_chunks v
                JOIN chunks c ON c.rowid = v.rowid
                WHERE c.session_id = ?
                LIMIT 10
            )""",
            (sid,),
        ).fetchone()

        # sqlite-vec doesn't support AVG, so get individual vectors and average
        rows = conn.execute(
            """SELECT v.rowid FROM chunks c
               JOIN vec_chunks v ON v.rowid = c.rowid
               WHERE c.session_id = ?
               LIMIT 10""",
            (sid,),
        ).fetchall()

        if not rows:
            continue

        import struct
        vecs = []
        for (rid,) in rows:
            blob = conn.execute(
                "SELECT embedding FROM vec_chunks WHERE rowid = ?", (rid,)
            ).fetchone()
            if blob and blob[0]:
                vec = struct.unpack(f"{384}f", blob[0])
                vecs.append(vec)

        if vecs:
            avg_vec = np.mean(vecs, axis=0)
            embeddings.append(avg_vec)
            valid_session_ids.append(sid)

    if len(embeddings) < 5:
        # Too few for clustering, fall back to LLM
        return _discover_topics_llm(project_path, session_summaries)

    X = np.array(embeddings)

    # HDBSCAN — finds natural clusters without specifying K
    clusterer = HDBSCAN(min_cluster_size=3, min_samples=2)
    labels = clusterer.fit_predict(X)

    # Group sessions by cluster
    clusters: dict[int, list[str]] = {}
    for sid, label in zip(valid_session_ids, labels):
        if label == -1:
            # Noise — put in an "other" cluster
            clusters.setdefault(-1, []).append(sid)
        else:
            clusters.setdefault(label, []).append(sid)

    # Summarize each cluster with LLM
    topics = []
    project_name = Path(project_path).name if project_path else "unknown"

    for label, cluster_sids in sorted(clusters.items()):
        cluster_summaries = [
            summary_by_id[sid] for sid in cluster_sids
            if sid in summary_by_id
        ]
        if not cluster_summaries:
            continue

        sessions_block = "\n\n".join(
            f"[{s['session_id'][:8]}] [{s['timestamp'][:10]}] {s['summary'][:400]}"
            for s in cluster_summaries
        )
        if len(sessions_block) > MAX_INPUT_CHARS:
            sessions_block = sessions_block[:MAX_INPUT_CHARS]

        prompt = f"""You are summarizing a group of related Claude Code sessions from "{project_name}".

SESSIONS IN THIS GROUP:
{sessions_block}

These sessions were grouped together because they share a common theme or workstream. Identify what that theme is and summarize it.

Return JSON with:
- "topic_name": Short descriptive name for this workstream
- "status": Current status in 1 sentence
- "narrative": What this workstream covers, what's been done, where it stands (3-5 sentences)
- "key_decisions": Important decisions (up to 3)
- "open_threads": Unresolved items (up to 3)

Return raw JSON only."""

        result = _call_claude(prompt, CLUSTER_SUMMARY_SCHEMA)
        if result:
            result["session_ids"] = [sid[:8] for sid in cluster_sids]
            topics.append(result)

    return topics if topics else None


def _discover_topics_llm_batched(
    project_path: str,
    session_summaries: list[dict],
) -> list[dict] | None:
    """Fallback for large projects without HDBSCAN: batch LLM calls.

    Split sessions into time-based batches, discover topics per batch,
    then merge similar topics in a final pass.
    """
    batch_size = 80  # fits in one context window
    all_topics = []

    for i in range(0, len(session_summaries), batch_size):
        batch = session_summaries[i:i + batch_size]
        topics = _discover_topics_llm(project_path, batch)
        if topics:
            all_topics.extend(topics)

    if not all_topics:
        return None

    # If only one batch, no merge needed
    if len(session_summaries) <= batch_size:
        return all_topics

    # Merge pass: feed topic summaries back to LLM to consolidate
    topics_block = "\n\n".join(
        f"**{t.get('topic_name', '?')}**: {t.get('narrative', '')[:300]}"
        for t in all_topics
    )

    if len(topics_block) > MAX_INPUT_CHARS:
        topics_block = topics_block[:MAX_INPUT_CHARS]

    merge_prompt = f"""You discovered these topics across multiple batches of sessions from the same project. Some may be duplicates or closely related.

TOPICS FOUND:
{topics_block}

Merge duplicates and closely related topics. Return the consolidated list.

Return JSON with a "topics" array, each with:
- "topic_name", "status", "narrative", "key_decisions", "open_threads", "session_ids"

Return raw JSON only."""

    result = _call_claude(merge_prompt, TOPIC_DISCOVERY_SCHEMA, timeout=120)
    if result:
        return result.get("topics", all_topics)
    return all_topics


# ── Portfolio (L3) ───────────────────────────────────────────

def summarize_portfolio(project_summaries: list[dict]) -> dict | None:
    """Generate an L3 portfolio rollup from L2 topic summaries."""
    if not project_summaries:
        return None

    projects_block = "\n\n".join(
        f"**{p.get('name', 'unknown')}** ({p.get('path', '')})\n{p.get('summary_text', '')[:600]}"
        for p in project_summaries
    )

    if len(projects_block) > MAX_INPUT_CHARS:
        projects_block = projects_block[:MAX_INPUT_CHARS]

    prompt = f"""You are creating a portfolio rollup across multiple active projects.

PROJECTS ({len(project_summaries)} total):
{projects_block}

Synthesize into a portfolio view. What's moving, what's stalled, what needs attention?

Return JSON with:
- "title": Portfolio title (e.g., "AIC Holdings — Week of March 30")
- "summary": 2-3 sentence overview of activity across all projects
- "active_projects": Array of objects with "name", "status" (1 sentence), "one_liner" (what's happening)
- "stalled_projects": Array of project names with no recent activity
- "themes": Array of cross-cutting themes or patterns (up to 3)

Return raw JSON only."""

    return _call_claude(prompt, PORTFOLIO_SCHEMA)


# ── Main entry point ─────────────────────────────────────────

def discover_and_summarize_topics(
    project_path: str,
    conn,
    cache: SessionCache | None = None,
) -> list[dict] | None:
    """Full pipeline: discover topics within a project and generate L2 summaries.

    Every project gets topic discovery regardless of size.
    - Small (< 100 sessions): LLM groups + summarizes in one call
    - Large (≥ 100 sessions): HDBSCAN pre-clusters, LLM summarizes each

    Returns list of topic dicts, or None on failure.
    Stores each topic as a separate L2 summary in summary_levels.
    """
    from .insights import upsert_summary

    session_summaries = _gather_session_summaries(project_path, conn, cache)
    if not session_summaries:
        return None

    # Discover topics
    if len(session_summaries) < LLM_DIRECT_THRESHOLD:
        topics = _discover_topics_llm(project_path, session_summaries)
    else:
        topics = _discover_topics_clustered(project_path, session_summaries, conn)

    if not topics:
        return None

    # Store each topic as a separate L2 summary
    project_name = Path(project_path).name if project_path else "unknown"

    for topic in topics:
        topic_name = topic.get("topic_name", "Unknown Topic")
        entity_id = f"{project_path}::{topic_name}"

        # Collect source session IDs
        source_ids = topic.get("session_ids", [])

        upsert_summary(
            conn,
            level=2,
            entity_id=entity_id,
            entity_type="project",
            title=f"{project_name}: {topic_name}",
            summary_text=json.dumps(topic),
            source_ids=source_ids,
        )

    return topics


# ── Legacy compat ────────────────────────────────────────────

def generate_project_summary(
    project_path: str,
    conn,
    cache: SessionCache | None = None,
) -> dict | None:
    """Legacy entry point — now delegates to topic discovery.

    Returns the first/primary topic summary for backward compatibility.
    """
    topics = discover_and_summarize_topics(project_path, conn, cache)
    if not topics:
        return None
    # Return first topic for backward compat, but all are stored
    return topics[0]
