"""Re-indexing CLI for insights database.

Detects stale embeddings (model version mismatch), orphaned chunks
(session file deleted), and re-processes them. Idempotent — safe to
run repeatedly.

Entry point: claude-session-reindex
"""

import argparse
import sys
import time
from pathlib import Path

from .paths import decode_project_path

CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"


def _find_session_file(session_id: str) -> Path | None:
    """Find the JSONL file for a session ID across all project dirs."""
    if not PROJECTS_DIR.exists():
        return None
    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        candidate = project_dir / f"{session_id}.jsonl"
        if candidate.exists():
            return candidate
    return None


def reindex(
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Re-index insights database.

    Returns dict with counts: {reindexed, orphans_removed, errors, skipped}.
    """
    from . import insights as ins
    from .git_context import get_git_anchor

    db_path = Path(ins.DB_PATH)
    if not db_path.exists():
        return {"error": "insights.db not found", "reindexed": 0, "orphans_removed": 0}

    conn = ins.get_db(str(db_path))
    ins.init_db(conn)

    # Load embedding model
    try:
        from fastembed import TextEmbedding
        model = TextEmbedding(ins.EMBEDDING_MODEL)
    except ImportError:
        return {"error": "fastembed not installed", "reindexed": 0, "orphans_removed": 0}

    # Get all indexed sessions
    sessions = conn.execute(
        "SELECT DISTINCT session_id FROM chunks"
    ).fetchall()

    stats = {"reindexed": 0, "orphans_removed": 0, "errors": 0, "skipped": 0}
    total = len(sessions)

    for i, (session_id,) in enumerate(sessions):
        if verbose:
            print(f"[{i+1}/{total}] {session_id[:12]}...", end=" ", flush=True)

        # Find the source file
        session_file = _find_session_file(session_id)

        if session_file is None:
            # Orphaned — session file no longer exists
            if dry_run:
                if verbose:
                    print("ORPHAN (would remove)")
                stats["orphans_removed"] += 1
                continue

            ins._delete_session_chunks(conn, session_id)
            conn.commit()
            if verbose:
                print("ORPHAN removed")
            stats["orphans_removed"] += 1
            continue

        if not force:
            if verbose:
                print("skip (use --force to re-embed)")
            stats["skipped"] += 1
            continue

        # Re-index
        if dry_run:
            if verbose:
                print("WOULD reindex")
            stats["reindexed"] += 1
            continue

        try:
            project_path = decode_project_path(session_file.parent.name)
            git_anchor = get_git_anchor(project_path)

            ins._delete_session_chunks(conn, session_id)
            conn.commit()

            turns, subagents = ins.index_session(
                str(session_file), conn, model=model,
                session_id=session_id,
                project_path=project_path,
                git_anchor=git_anchor if git_anchor.get("branch") else None,
            )
            if verbose:
                print(f"OK ({turns} turns, {subagents} subagents)")
            stats["reindexed"] += 1
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            stats["errors"] += 1

    conn.close()
    return stats


def backfill(
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Index sessions that exist on disk but are missing from insights DB.

    Discovers all sessions via find_all_sessions(), checks which are NOT
    in the insights DB, and indexes them. This catches sessions that were
    cached by the daemon but never made it into the semantic search index.

    Returns dict with counts: {indexed, errors, skipped, total_found}.
    """
    from . import insights as ins
    from .discovery import find_all_sessions
    from .git_context import get_git_anchor

    db_path = Path(ins.DB_PATH)
    if not db_path.exists():
        return {"error": "insights.db not found", "indexed": 0, "errors": 0}

    conn = ins.get_db(str(db_path))
    ins.init_db(conn)

    # Load embedding model
    try:
        from fastembed import TextEmbedding
        model = TextEmbedding(ins.EMBEDDING_MODEL)
    except ImportError:
        return {"error": "fastembed not installed", "indexed": 0, "errors": 0}

    # Load skip list for known-bad sessions
    skip_file = CLAUDE_DIR / "insights-skip.json"
    skip_set = set()
    if skip_file.exists():
        try:
            import json
            skip_set = set(json.loads(skip_file.read_text()).get("skip", []))
            if verbose and skip_set:
                print(f"Skipping {len(skip_set)} known-bad sessions")
        except Exception:
            pass

    # Find all sessions on disk
    all_sessions = find_all_sessions()
    if verbose:
        print(f"Found {len(all_sessions)} sessions on disk")

    # Get already-indexed session IDs
    indexed_ids = set(
        r[0] for r in conn.execute("SELECT DISTINCT session_id FROM chunks").fetchall()
    )
    if verbose:
        print(f"Already indexed: {len(indexed_ids)}")

    # Find the gap
    to_index = [
        s for s in all_sessions
        if s["session_id"] not in indexed_ids
        and s["session_id"] not in skip_set
    ]
    if verbose:
        print(f"To index: {len(to_index)}")

    stats = {"indexed": 0, "errors": 0, "skipped": len(all_sessions) - len(to_index), "total_found": len(all_sessions)}

    for i, s in enumerate(to_index):
        sid = s["session_id"]
        if verbose:
            print(f"[{i+1}/{len(to_index)}] {sid[:12]}...", end=" ", flush=True)

        if dry_run:
            if verbose:
                print("WOULD index")
            stats["indexed"] += 1
            continue

        try:
            project_path = s["project_dir"]
            git_anchor = get_git_anchor(project_path)

            turns, subagents = ins.index_session(
                str(s["file"]), conn, model=model,
                session_id=sid,
                project_path=project_path,
                git_anchor=git_anchor if git_anchor.get("branch") else None,
            )
            if verbose:
                print(f"OK ({turns} turns, {subagents} subagents)")
            stats["indexed"] += 1
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            stats["errors"] += 1

    conn.close()
    return stats


def rebuild_fts(verbose: bool = False) -> bool:
    """Rebuild the FTS5 index from existing chunks data.

    Useful for existing databases that don't have FTS5 content yet.
    Returns True on success.
    """
    from . import insights as ins

    db_path = Path(ins.DB_PATH)
    if not db_path.exists():
        print("Error: insights.db not found")
        return False

    conn = ins.get_db(str(db_path))
    ins.init_db(conn)  # Ensures FTS5 table and triggers exist

    try:
        if verbose:
            print("Rebuilding FTS5 index...")
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()
        if verbose:
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            print(f"FTS5 index rebuilt with {count} chunks")
        return True
    except Exception as e:
        print(f"FTS rebuild error: {e}")
        return False
    finally:
        conn.close()


def main():
    """CLI entry point: claude-session-reindex."""
    parser = argparse.ArgumentParser(
        description="Re-index insights database (clean orphans, re-embed stale chunks)"
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-embed all sessions (not just stale ones)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without making changes")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress for each session")
    parser.add_argument("--rebuild-fts", action="store_true",
                        help="Rebuild FTS5 full-text search index from existing chunks")
    parser.add_argument("--backfill", action="store_true",
                        help="Index sessions that exist on disk but are missing from insights DB")
    args = parser.parse_args()

    print("claude-session-reindex")

    if args.rebuild_fts:
        success = rebuild_fts(verbose=args.verbose)
        sys.exit(0 if success else 1)

    if args.backfill:
        if args.dry_run:
            print("  DRY RUN — no changes will be made")
        start = time.time()
        result = backfill(dry_run=args.dry_run, verbose=args.verbose)
        elapsed = time.time() - start

        if "error" in result:
            print(f"\nError: {result['error']}")
            sys.exit(1)

        print(f"\nDone in {elapsed:.1f}s:")
        print(f"  Indexed:   {result['indexed']}")
        print(f"  Errors:    {result['errors']}")
        print(f"  Skipped:   {result['skipped']} (already indexed)")
        print(f"  Total:     {result['total_found']} sessions on disk")
        sys.exit(0)

    if args.dry_run:
        print("  DRY RUN — no changes will be made")

    start = time.time()
    result = reindex(force=args.force, dry_run=args.dry_run, verbose=args.verbose)
    elapsed = time.time() - start

    if "error" in result:
        print(f"\nError: {result['error']}")
        sys.exit(1)

    print(f"\nDone in {elapsed:.1f}s:")
    print(f"  Reindexed:       {result['reindexed']}")
    print(f"  Orphans removed: {result['orphans_removed']}")
    print(f"  Errors:          {result['errors']}")
    print(f"  Skipped:         {result.get('skipped', 0)}")


if __name__ == "__main__":
    main()
