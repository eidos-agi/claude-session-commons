"""CLI entry point for claude-find — semantic search over session transcripts.

Usage:
    claude-find "what mistakes were made with Navusoft?"
    claude-find "how did we handle WAM integration?" --limit 5
    claude-find "user corrections about email format" --type turn
    claude-find --stats
"""

import argparse
import sys

from .insights import get_db, init_db, query, get_stats, DB_PATH
from .paths import shorten_path


def _format_result(r, max_content: int = 300) -> str:
    """Format a single ChunkResult for terminal output."""
    # Parse timestamp to just date + time
    ts = r.timestamp
    if "T" in ts:
        ts = ts.replace("T", " ")[:16]

    project = shorten_path(r.project_path)
    header = f"[{r.distance:.2f}] {ts}  |  {r.chunk_type}  |  {project}"

    # Indent content, truncate
    content = r.content.strip()
    if len(content) > max_content:
        content = content[:max_content] + "..."

    lines = content.split("\n")
    indented = "\n".join(f"  {line}" for line in lines)

    return f"{header}\n{indented}"


def main():
    parser = argparse.ArgumentParser(
        prog="claude-find",
        description="Semantic search over Claude Code session transcripts",
    )
    parser.add_argument(
        "query", nargs="?",
        help="Natural language search query",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=10,
        help="Max results (default: 10)",
    )
    parser.add_argument(
        "--type", "-t", choices=["turn", "subagent_summary"],
        help="Filter by chunk type",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show index statistics",
    )
    parser.add_argument(
        "--db", default=DB_PATH,
        help=f"Database path (default: {DB_PATH})",
    )

    args = parser.parse_args()

    try:
        conn = get_db(args.db)
        init_db(conn)
    except Exception as e:
        print(f"Error opening database: {e}", file=sys.stderr)
        print("Install insights dependencies: pip install -e '.[insights]'", file=sys.stderr)
        sys.exit(1)

    if args.stats:
        stats = get_stats(conn)
        print(f"Sessions indexed: {stats['sessions_indexed']}")
        print(f"Total chunks:     {stats['total_chunks']}")
        print(f"  Turns:          {stats['turns']}")
        print(f"  Subagents:      {stats['subagent_summaries']}")
        conn.close()
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    results = query(
        args.query,
        conn,
        limit=args.limit,
        chunk_type=args.type,
    )

    if not results:
        print("No results found.")
        print(f"(Database has {get_stats(conn)['total_chunks']} chunks indexed)")
        conn.close()
        sys.exit(0)

    for i, r in enumerate(results):
        if i > 0:
            print()
        print(_format_result(r))

    conn.close()


if __name__ == "__main__":
    main()
