"""Playbook data model — structured templates for recurring deliverables.

Playbooks define reusable templates with sections, data-gathering steps,
and output format for recurring deliverables like executive briefs,
status reports, and project plans.
"""

import dataclasses
import json
import re
import uuid
from datetime import datetime, timezone

# Use pysqlite3 for extension loading support
try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3


@dataclasses.dataclass
class Playbook:
    """A structured template for a recurring deliverable."""

    id: str
    title: str
    description: str
    sections: list[dict]       # [{name, description, prompt_hint}, ...]
    data_steps: list[dict]     # [{tool, args_template, purpose}, ...]
    keywords: list[str]
    example_output: str        # Full example of what good output looks like
    intent_patterns: list[str] # Regex patterns for auto-matching user questions
    created_at: str
    updated_at: str


# ── CRUD functions ────────────────────────────────────────────

def list_playbooks(conn: sqlite3.Connection) -> list[Playbook]:
    """Return all non-deleted playbooks."""
    rows = conn.execute(
        """SELECT id, title, description, sections, data_steps, keywords,
                  example_output, intent_patterns, created_at, updated_at
           FROM playbooks
           WHERE deleted_at IS NULL
           ORDER BY title"""
    ).fetchall()
    return [_row_to_playbook(r) for r in rows]


def get_playbook(conn: sqlite3.Connection, playbook_id: str) -> Playbook | None:
    """Get one playbook by ID."""
    row = conn.execute(
        """SELECT id, title, description, sections, data_steps, keywords,
                  example_output, intent_patterns, created_at, updated_at
           FROM playbooks
           WHERE id = ? AND deleted_at IS NULL""",
        (playbook_id,),
    ).fetchone()
    return _row_to_playbook(row) if row else None


def search_playbooks(conn: sqlite3.Connection, query: str) -> list[Playbook]:
    """Search by title, description, keywords (LIKE match)."""
    pattern = f"%{query}%"
    rows = conn.execute(
        """SELECT id, title, description, sections, data_steps, keywords,
                  example_output, intent_patterns, created_at, updated_at
           FROM playbooks
           WHERE deleted_at IS NULL
             AND (title LIKE ? OR description LIKE ? OR keywords LIKE ?)
           ORDER BY title""",
        (pattern, pattern, pattern),
    ).fetchall()
    return [_row_to_playbook(r) for r in rows]


def create_playbook(
    conn: sqlite3.Connection,
    title: str,
    description: str,
    sections: list[dict],
    data_steps: list[dict] | None = None,
    keywords: list[str] | None = None,
    example_output: str = "",
    intent_patterns: list[str] | None = None,
    playbook_id: str | None = None,
) -> Playbook:
    """Insert a new playbook. Returns the created Playbook."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    pid = playbook_id or str(uuid.uuid4())[:8]
    conn.execute(
        """INSERT INTO playbooks (id, title, description, sections, data_steps,
                                   keywords, example_output, intent_patterns,
                                   created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pid,
            title,
            description,
            json.dumps(sections),
            json.dumps(data_steps or []),
            json.dumps(keywords or []),
            example_output,
            json.dumps(intent_patterns or []),
            now,
            now,
        ),
    )
    conn.commit()
    return get_playbook(conn, pid)


def update_playbook(
    conn: sqlite3.Connection,
    playbook_id: str,
    title: str | None = None,
    description: str | None = None,
    sections: list[dict] | None = None,
    data_steps: list[dict] | None = None,
    keywords: list[str] | None = None,
    example_output: str | None = None,
    intent_patterns: list[str] | None = None,
) -> Playbook | None:
    """Update fields on an existing playbook. Returns updated Playbook."""
    existing = get_playbook(conn, playbook_id)
    if not existing:
        return None

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    conn.execute(
        """UPDATE playbooks
           SET title = ?, description = ?, sections = ?, data_steps = ?,
               keywords = ?, example_output = ?, intent_patterns = ?,
               updated_at = ?
           WHERE id = ? AND deleted_at IS NULL""",
        (
            title if title is not None else existing.title,
            description if description is not None else existing.description,
            json.dumps(sections) if sections is not None else json.dumps(existing.sections),
            json.dumps(data_steps) if data_steps is not None else json.dumps(existing.data_steps),
            json.dumps(keywords) if keywords is not None else json.dumps(existing.keywords),
            example_output if example_output is not None else existing.example_output,
            json.dumps(intent_patterns) if intent_patterns is not None else json.dumps(existing.intent_patterns),
            now,
            playbook_id,
        ),
    )
    conn.commit()
    return get_playbook(conn, playbook_id)


def delete_playbook(conn: sqlite3.Connection, playbook_id: str) -> bool:
    """Soft delete a playbook. Returns True if found and deleted."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    cursor = conn.execute(
        """UPDATE playbooks SET deleted_at = ?
           WHERE id = ? AND deleted_at IS NULL""",
        (now, playbook_id),
    )
    conn.commit()
    return cursor.rowcount > 0


# ── Helpers ───────────────────────────────────────────────────

def _row_to_playbook(row: tuple) -> Playbook:
    return Playbook(
        id=row[0],
        title=row[1],
        description=row[2],
        sections=json.loads(row[3]),
        data_steps=json.loads(row[4]),
        keywords=json.loads(row[5]),
        example_output=row[6] or "",
        intent_patterns=json.loads(row[7]) if row[7] else [],
        created_at=row[8],
        updated_at=row[9],
    )


# ── Auto-matching ─────────────────────────────────────────────

def match_playbook_by_intent(conn: sqlite3.Connection, message: str) -> Playbook | None:
    """Fast-path: test message against playbook intent_patterns (regex).

    Returns the first matching playbook, or None.
    """
    playbooks = list_playbooks(conn)
    for pb in playbooks:
        if not pb.intent_patterns:
            continue
        for pattern in pb.intent_patterns:
            try:
                if re.search(pattern, message):
                    return pb
            except re.error:
                continue
    return None


def match_playbook_semantic(
    conn: sqlite3.Connection,
    message: str,
    model,
    threshold: float = 0.60,
) -> Playbook | None:
    """Semantic match: embed the user message and compare to playbook descriptions.

    Uses cosine similarity between the message embedding and each playbook's
    description + keywords text. Returns the best match above threshold.

    The threshold is tuned for bge-small-en-v1.5 cosine similarity where
    0.60+ indicates genuine topical relevance. Generic messages ("hello",
    "what time is it") score 0.50-0.57, while real matches score 0.60+.
    """
    if model is None:
        return None

    playbooks = list_playbooks(conn)
    if not playbooks:
        return None

    # Build comparison texts: description + keywords for each playbook
    pb_texts = []
    for pb in playbooks:
        kw_str = ", ".join(pb.keywords)
        section_names = ", ".join(s["name"] for s in pb.sections)
        pb_texts.append(f"{pb.title}. {pb.description} Keywords: {kw_str}. Sections: {section_names}")

    # Embed all in one batch + the user message
    all_texts = [message] + pb_texts
    try:
        embeddings = list(model.embed(all_texts))
    except Exception:
        return None

    msg_vec = embeddings[0]

    # Cosine similarity against each playbook
    best_score = -1.0
    best_pb = None
    for i, pb in enumerate(playbooks):
        pb_vec = embeddings[i + 1]
        score = _cosine_sim(msg_vec, pb_vec)
        if score > best_score:
            best_score = score
            best_pb = pb

    if best_score >= threshold and best_pb is not None:
        return best_pb
    return None


def match_playbook(
    conn: sqlite3.Connection,
    message: str,
    model=None,
) -> Playbook | None:
    """Find the best playbook for a user message.

    Tries intent patterns first (fast, microseconds), then falls back
    to semantic matching (100-200ms). Returns None if no match.
    """
    # Fast path: regex intent patterns
    matched = match_playbook_by_intent(conn, message)
    if matched:
        return matched

    # Slow path: semantic similarity
    if model is not None:
        return match_playbook_semantic(conn, message, model)

    return None


def _cosine_sim(a, b) -> float:
    """Cosine similarity between two vectors (numpy arrays or lists)."""
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ── Seed defaults ─────────────────────────────────────────────

EXECUTIVE_BRIEF = {
    "id": "executive-brief",
    "title": "Executive Brief",
    "description": "Comprehensive project status brief for leadership. EVERY section must be written with substantive detail (3-8 sentences minimum per section). This is NOT a quick summary — it is a thorough briefing document.",
    "sections": [
        {
            "name": "Executive Summary",
            "description": "3-5 sentence high-level overview covering the project's current health, most important recent accomplishment, and top priority going forward. Written for a busy executive who reads only this section.",
            "prompt_hint": "Write 3-5 complete sentences. Cover: overall health/trajectory, the single biggest accomplishment, and what matters most right now. Do NOT just list bullet points — write prose.",
        },
        {
            "name": "Where We Were",
            "description": "Describe the baseline state before recent work began. What was the starting point? What existed? What didn't? What problems or gaps needed addressing? Include specific dates, versions, or states where known. Minimum 3-5 sentences.",
            "prompt_hint": "Use session history to reconstruct what the project looked like 2-4 weeks ago. Reference specific capabilities that existed or didn't, infrastructure state, team setup. Be specific with dates and details.",
        },
        {
            "name": "Where We Are",
            "description": "Comprehensive description of current state — what has been accomplished, what is actively in progress, what is working. Reference specific commits, shipped features, and infrastructure changes. This should be the longest section. Minimum 5-10 sentences.",
            "prompt_hint": "This is the MEAT of the brief. Use commit history to enumerate specific things that shipped. Group by area (infrastructure, features, documentation, team, etc.). Include commit dates. Don't just list — explain the significance of each accomplishment. Be thorough and detailed.",
        },
        {
            "name": "Where We Are Going",
            "description": "Future direction — next milestones, strategic goals, planned work for the next 2-4 weeks. What needs to happen next and why? Minimum 4-6 sentences.",
            "prompt_hint": "Outline the roadmap. What work is planned? What are the strategic priorities? What will the project look like in 30 days if everything goes well? Reference any stated goals from sessions.",
        },
        {
            "name": "Key Decisions Made",
            "description": "Important decisions since last update. For each decision: what was decided, the reasoning/context, and implications. These are the choices that shaped the project's direction. Minimum 3-5 bullet points with explanation.",
            "prompt_hint": "Extract decisions from session transcripts and commit messages. Format as a numbered list. Each decision should have: the decision itself (bold), the reasoning, and any implications. Don't just state the decision — explain WHY it was made.",
        },
        {
            "name": "Wins & Accomplishments",
            "description": "Notable achievements, shipped features, completed milestones. Celebrate what went well. Include specific deliverables with dates. Minimum 5-8 bullet points.",
            "prompt_hint": "List concrete deliverables from commit history and sessions. Each item should reference what was built, when, and why it matters. Group by category if there are many. Include metrics where available (e.g., '46 logo variations', '29-file knowledge base').",
        },
        {
            "name": "Assignments",
            "description": "Who owns what — every named person with their specific responsibilities, current tasks, and status. Format as a clear table or structured list. Include anyone mentioned in sessions or commits.",
            "prompt_hint": "Create a structured list of every person involved. For each person: Name, Role/Title if known, Current Assignments (bulleted), Status. If ownership is unclear for something, flag it as 'Unassigned'. Be thorough — look through all sessions and commits for names.",
        },
        {
            "name": "Blockers",
            "description": "Issues preventing progress — what is stuck, what is needed to unblock each item, and who owns the resolution. If no blockers, state 'No active blockers' and explain why (this is a positive signal worth noting). Minimum 2-3 sentences even if clear.",
            "prompt_hint": "Search for anything described as blocked, waiting, stalled, or dependent on external factors. For each blocker: describe the issue, its impact, what's needed to resolve it, and who owns it. If genuinely no blockers, say so clearly and note what's enabling smooth progress.",
        },
        {
            "name": "Risks & Mitigations",
            "description": "Potential future issues with assessment of likelihood and impact, plus what is being done to mitigate each risk. Every project has risks — identify at least 2-3 even if things are going well. Think about: timeline risk, dependency risk, technical risk, team/capacity risk, scope risk.",
            "prompt_hint": "Format as a table or structured list with columns: Risk, Likelihood (High/Med/Low), Impact (High/Med/Low), Mitigation. Think critically — even healthy projects have risks. Consider: key person dependencies, external API integrations, timeline pressure, scope creep, technical debt. Minimum 3 risks.",
        },
        {
            "name": "Timeline & Milestones",
            "description": "Chronological view of key dates — completed milestones with actual dates, upcoming deadlines, and any slippage from original plans. Format as a timeline. Minimum 5-8 entries.",
            "prompt_hint": "Build a timeline from commit dates, session timestamps, and any stated deadlines. Format: [Date] — Milestone/Event. Include both past (completed) and future (planned) items. If there's slippage, note original vs actual dates. End with the next 2-3 upcoming milestones.",
        },
    ],
    "data_steps": [
        {
            "tool": "hybrid_search",
            "args_template": {"query": "{project_name} progress status"},
            "purpose": "Find session context about project progress",
        },
        {
            "tool": "github_list_repos",
            "args_template": {"org": "{org}"},
            "purpose": "Discover all repos in the org",
        },
        {
            "tool": "github_search_commits",
            "args_template": {"query": "{project_name}", "org": "{org}"},
            "purpose": "Find what was actually shipped",
        },
        {
            "tool": "hybrid_search",
            "args_template": {"query": "{project_name} decisions blockers risks"},
            "purpose": "Find decisions, blockers, and risk discussions",
        },
    ],
    "keywords": ["executive", "brief", "status", "progress", "report", "summary", "leadership"],
    "intent_patterns": [
        r"(?i)(write|create|prepare|draft|generate)\s+(an?\s+)?(executive|leadership)\s+(brief|report|summary)",
        r"(?i)(comprehensive|full|detailed)\s+(status\s+)?(report|brief|update)\s+for\s+(leadership|executives|board|management)",
    ],
    "example_output": """# Executive Brief — Project Acme
**Date:** February 21, 2026 | **Prepared by:** Session Intelligence Agent

**Health: ON TRACK** | Velocity: High | Team: 3/4 staffed | Beta: March 15th

---

## Executive Summary

Project Acme has transitioned from initial setup into active feature development over the past three weeks, with strong shipping velocity across infrastructure, API, and frontend surfaces. The team completed 47 commits across 3 repositories since February 1st, establishing the core data pipeline and launching the first customer-facing dashboard. The primary focus now shifts to API integration with two external vendors (Navusoft and WidgetCorp) which are critical-path for the March 15th beta milestone. Two items need leadership attention: a Navusoft vendor credential delay that risks the timeline, and an unassigned QA role that needs staffing by Feb 28th.

**Needs from leadership:**
- Escalation path for Navusoft vendor delay (see Blockers)
- QA resource staffing decision by Feb 28th
- Confirmation of 3 beta customer selections for March onboarding

---

## Where We Were

As of early February 2026, Project Acme existed as a set of design documents and a bare repository scaffold. The team had completed initial architecture reviews and selected the technology stack (FastAPI + PostgreSQL + React), but no production code had been written. The Navusoft API documentation had been received but not yet analyzed. The team consisted of three engineers (Daniel, Sarah, Marcus) with no formal project management tooling in place. CI/CD was not configured, and there was no staging environment. The original project timeline called for a beta launch by March 1st, which has since been revised to March 15th based on the scope of vendor API integration work discovered during implementation.

---

## Where We Are

**Infrastructure & DevOps:**
The project now has a fully operational CI/CD pipeline on Railway with automated deployments from the `main` branch. A staging environment was provisioned on February 8th (`acme-staging.railway.app`) with production following on February 12th. Database migrations are managed through Alembic with 14 migration files tracking schema evolution. Monitoring is handled through a custom health check endpoint (`/api/health`) that reports database connectivity, cache status, and external API reachability.

**Backend API:**
The FastAPI backend has 23 endpoints across 4 route modules (auth, customers, data-pipeline, reports). Authentication uses JWT tokens with a 24-hour expiry and refresh token rotation. The data pipeline module processes incoming CSV uploads, validates against the Acme schema, and stores normalized records in PostgreSQL. Batch processing supports files up to 50MB with progress tracking via WebSocket.

**Frontend:**
The React dashboard (built with Vite + TailwindCSS) provides customer management, data upload, and a reporting view with 6 chart types. The upload flow includes drag-and-drop with real-time progress bars. The reporting page supports date range filtering and CSV export.

**Data:**
12,400 test records have been loaded from the Navusoft sample dataset. Data quality validation catches 4 known edge cases (missing ZIP codes, duplicate customer IDs, malformed dates, and negative quantities).

---

## Where We Are Going

The next 30 days focus on three areas: (1) completing the Navusoft real-time API integration, which will replace the current CSV upload flow for production data ingestion, (2) building the WidgetCorp inventory sync module, and (3) onboarding 3 beta customers for the March 15th soft launch. After beta, the team plans to add automated alerting, a customer-facing API, and multi-tenant support. The long-term vision is a self-service platform where customers can connect their own data sources, configure dashboards, and set up automated reports without engineering involvement.

---

## Key Decisions Made

1. **Revised beta timeline from March 1st to March 15th** — The Navusoft API turned out to require OAuth2 with PKCE flow rather than simple API key auth, adding approximately 1 week of work. Decision made jointly by Daniel and Sarah on Feb 10th after the API analysis session. Impact: 2-week slip on beta, but higher quality integration.

2. **PostgreSQL over DynamoDB for primary storage** — Despite initial consideration of DynamoDB for cost at scale, the team chose PostgreSQL for its relational query capabilities needed by the reporting engine. This was decided on Feb 5th based on Marcus's prototype benchmarks showing complex report queries were 4x faster. Impact: ~$200/mo higher hosting cost, but dramatically simpler reporting layer.

3. **Railway over AWS for hosting** — Chose Railway for faster deployment iteration during the beta phase, with a plan to evaluate AWS migration post-launch if scale requires it. Decided Feb 3rd. Impact: deployment time from 15min (AWS) to 3min (Railway), saving ~2hrs/week during active development.

4. **JWT auth with refresh tokens over session-based auth** — Selected for better mobile app compatibility (planned for Q2). Decided Feb 7th.

5. **Soft deletes for all customer data** — Regulatory compliance requires 90-day data retention. All DELETE operations use `deleted_at` timestamps. Decided Feb 4th. Impact: slightly more complex queries, but audit-ready from day one.

---

## Wins & Accomplishments

- **47 commits shipped** across `acme-api`, `acme-web`, and `acme-infra` repos (Feb 1-21) — averaging 2.2 commits/day
- **CI/CD pipeline operational** — zero-downtime deployments from merge to production in under 3 minutes
- **Staging environment live** since Feb 8th — used for all QA before production pushes
- **14 database migrations** — clean schema evolution with zero data loss across 3 weeks
- **23 API endpoints** — full CRUD for customers, data pipeline, and reports
- **Dashboard MVP shipped** — 6 chart types, date filtering, CSV export all functional and demo-ready
- **12,400 test records loaded** — validated against 4 known edge cases, giving confidence in data pipeline correctness
- **Navusoft API analysis complete** — documented all 34 endpoints, identified 12 needed for MVP, surfaced the OAuth2 complexity early (before it became a surprise)
- **Team provisioned on Claude Teams** — all 3 engineers have AI-assisted development, measurably faster code review and documentation

---

## Assignments

| Person | Role | Current Focus | Status |
|--------|------|---------------|--------|
| **Daniel** | Lead / Architecture | Navusoft OAuth2 integration, API design review, vendor escalation | Active — carrying vendor relationship + technical work |
| **Sarah** | Backend Engineer | Data pipeline batch processing, WidgetCorp API research | Active |
| **Marcus** | Frontend Engineer | Dashboard reporting views, chart performance optimization | Active |
| **TBD** | QA / Beta Coordination | Beta customer onboarding, test plan execution, regression suite | **Unassigned — need to identify by Feb 28th** |

**Note:** Daniel is currently the only person with Navusoft vendor relationship context AND the lead architect. This is a single-point-of-failure risk (see Risks).

---

## Blockers

1. **Navusoft sandbox credentials delayed** — Requested Feb 14th, still pending Navusoft support ticket #4821. Without sandbox access, OAuth2 integration can only be built against mocked responses. Daniel is following up daily; escalation to Navusoft account manager planned for Feb 24th if unresolved. **Leadership ask:** If unresolved by Feb 24th, can we use the Jetta Operations relationship to escalate?

2. **No QA resource assigned** — Beta launch requires systematic testing across data import, reporting, and edge cases. Currently handled ad-hoc by engineers, which slows feature development by an estimated 20-30%. **Leadership ask:** Approve QA staffing (contract or internal) by Feb 28th.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|-----------|--------|------------|-------|
| Navusoft sandbox access delays push past March 15th | **Medium** | **High** | Building against mocked API responses; can switch to real sandbox with <1 day work once credentials arrive. Escalation plan at Feb 24th. | Daniel |
| WidgetCorp API undocumented edge cases | Medium | Medium | Sarah doing exploratory testing with sample data this week; building comprehensive error handling with fallback modes | Sarah |
| Beta customer data volumes exceed test dataset | Low | Medium | Load tested to 50K records; PostgreSQL EXPLAIN plans reviewed for all report queries; can vertically scale Railway instance same-day | Marcus |
| **Key person risk — Daniel is sole architect AND vendor contact** | **Medium** | **High** | Sarah is ramping on architecture knowledge; all ADRs documented in `planning/adrs/`. Recommend Daniel record a 30-min architecture walkthrough video. | Daniel + Sarah |
| Scope creep from beta customer feedback | High | Medium | Strict "observe but don't promise" policy for beta; all feature requests go to backlog triage, not current sprint | Daniel |

---

## Timeline & Milestones

| Date | Milestone | Status |
|------|-----------|--------|
| Feb 1 | Project kickoff, repo scaffolding | Done |
| Feb 5 | Technology stack finalized (FastAPI + PostgreSQL + React) | Done |
| Feb 8 | Staging environment live on Railway | Done |
| Feb 12 | Production environment provisioned | Done |
| Feb 15 | Dashboard MVP with 6 chart types | Done |
| Feb 18 | Navusoft API analysis document complete | Done |
| Feb 21 | Data pipeline batch processing live | Done |
| **Feb 24** | **Navusoft escalation deadline if sandbox still pending** | Upcoming |
| **Feb 28** | **Navusoft OAuth2 integration complete** | In Progress |
| **Feb 28** | **QA resource identified and onboarded** | At Risk — unassigned |
| **Mar 5** | **WidgetCorp inventory sync MVP** | Planned |
| **Mar 10** | **Beta customer onboarding begins (3 customers)** | Planned |
| **Mar 15** | **Beta launch** | Planned — on track if Navusoft unblocks |
| Mar 31 | Beta feedback review & Q2 planning | Planned |

**Critical path:** Navusoft credentials → OAuth2 integration → Beta launch. If credentials arrive by Feb 24th, March 15th is achievable. If not, beta slips to ~March 22nd.
""",
}


PROJECT_STATUS_CHECK = {
    "id": "project-status",
    "title": "Project Status Check",
    "description": "Quick project status summary — recent activity, current state, and next steps. Lighter than an executive brief, focused on answering 'where are we?' and 'where did we leave off?' questions.",
    "sections": [
        {
            "name": "Current State",
            "description": "What is the project's current state? Active, paused, blocked? What phase is it in? What's working today? 2-4 sentences.",
            "prompt_hint": "Synthesize from recent sessions and commits. State the phase (setup, active dev, maintenance, etc.) and overall health. Be specific — name what's live.",
        },
        {
            "name": "Recent Activity",
            "description": "What happened in the last 1-2 weeks? Commits, session topics, key changes. Chronological, with dates. 5-10 bullet points.",
            "prompt_hint": "Use github_search_commits and hybrid_search. Include dates, commit SHAs, repo names, and brief descriptions. Group by repo if multiple repos are involved. Most recent first.",
        },
        {
            "name": "Key Changes Since Last Touch",
            "description": "What's different from when this was last actively discussed? New capabilities, architectural shifts, team changes. 3-5 bullet points.",
            "prompt_hint": "Compare recent activity to older sessions. Highlight what's new or changed. If you can identify a gap in activity (e.g., 'no commits for 5 days'), note it.",
        },
        {
            "name": "What's Next",
            "description": "Immediate next steps — what should happen in the next 1-2 weeks? Planned work, open questions, upcoming deadlines. 3-5 bullet points.",
            "prompt_hint": "Look for stated plans, TODOs, and open items in sessions. Be specific about what needs to happen, who owns it if known, and any deadlines.",
        },
        {
            "name": "Blockers & Concerns",
            "description": "Anything blocking progress or causing concern? Dependencies, missing resources, stalled work. If none, say so clearly — that's worth noting.",
            "prompt_hint": "Search for blocked/waiting/stalled mentions. If genuinely clear, state 'No active blockers' with one sentence on what's enabling smooth progress.",
        },
    ],
    "data_steps": [
        {
            "tool": "hybrid_search",
            "args_template": {"query": "{project_name} status progress recent"},
            "purpose": "Find session context about the project",
        },
        {
            "tool": "github_search_commits",
            "args_template": {"query": "{project_name}", "org": "{org}"},
            "purpose": "Find recent commits for this project",
        },
        {
            "tool": "hybrid_search",
            "args_template": {"query": "{project_name} blockers next steps plans"},
            "purpose": "Find blockers and planned work",
        },
    ],
    "keywords": ["status", "progress", "update", "check", "where", "state", "catch up", "left off", "how is"],
    "intent_patterns": [
        r"(?i)where\s+(are|did)\s+we\b.*\b(on|with|at|leave)",
        r"(?i)what('s|\s+is)\s+the\s+(status|state|progress)\s+(of|on|with)\b",
        r"(?i)catch\s+me\s+up\s+(on|with)\b",
        r"(?i)how('s|\s+is)\s+\w+\s+(going|progressing|coming\s+along)",
        r"(?i)where\s+did\s+we\s+leave\s+off",
        r"(?i)(status|progress)\s+(check|update)\s+(on|for)\b",
        r"(?i)bring\s+me\s+up\s+to\s+(speed|date)",
        r"(?i)give\s+me\s+(a\s+)?(quick\s+)?(status|update|rundown)\s+(on|for|about)\b",
    ],
    "example_output": """# Project Status: Cerebro
**As of:** February 21, 2026 | **Status: Active Development**

---

## Current State

Cerebro is in active development, mid-build on the executive and operations dashboards. The core infrastructure is in place — Railway deployment, PostgreSQL backend, and Greenmark brand system (green palette, Figtree font, card-based UI). The executive dashboard is functional with data visualization, and the operations view is being built out. The QA dashboard for Cerebro-specific data quality monitoring was recently added.

---

## Recent Activity

- **[2026-02-13]** `greenmark-waste-solutions/cerebro` — Latest push, dashboard iteration and brand refinements
- **[2026-02-12]** Added QA Dashboard for Cerebro data quality monitoring
- **[2026-02-10]** Executive dashboard cards — financial overview, operations metrics
- **[2026-02-08]** Brand implementation — Greenmark green (#203C31), Figtree font, standardized card layout
- **[2026-02-06]** Initial operations view with route/truck data visualization
- **[2026-02-04]** Database schema for operational metrics (medallion pattern from data-daemon)

---

## Key Changes Since Last Touch

- **QA infrastructure added** — Cerebro now has its own data quality dashboard, separate from the main exec view
- **Brand system locked in** — All components use the Greenmark palette and typography consistently
- **Data-daemon integration deepened** — Cerebro pulls from the medallion data warehouse rather than querying vendor APIs directly
- **Financial module started** — Cost accounting views are in progress, dependent on the research-cost-accounting repo

---

## What's Next

1. **Complete financial module** — Needs cost accounting model finalization from `research-cost-accounting` repo
2. **Operations dashboard build-out** — Route optimization view, truck utilization charts
3. **Mobile responsiveness pass** — Current UI is desktop-first, needs tablet/phone layouts for field use
4. **Beta deployment** — Target: internal stakeholders reviewing dashboards by end of February

---

## Blockers & Concerns

- **Cost accounting model not finalized** — The financial module in Cerebro is waiting on research outputs from `research-cost-accounting`. Not a hard blocker (other work continues) but it's the critical dependency for the financial dashboard.
- No other active blockers. Infrastructure is stable, data pipeline is feeding correctly.

**Quick stats:** 12 commits (Feb) | 1 repo | 2 contributors | last push: Feb 13
""",
}


def seed_defaults(conn: sqlite3.Connection):
    """Insert or update default playbooks.

    Uses INSERT OR REPLACE so seed content stays current when
    sections, descriptions, or example_output are improved.
    """
    for playbook_data in [EXECUTIVE_BRIEF, PROJECT_STATUS_CHECK]:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        # Check if exists — preserve created_at if updating
        existing = conn.execute(
            "SELECT created_at FROM playbooks WHERE id = ?",
            (playbook_data["id"],),
        ).fetchone()
        created = existing[0] if existing else now
        conn.execute(
            """INSERT OR REPLACE INTO playbooks
               (id, title, description, sections, data_steps,
                keywords, example_output, intent_patterns,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                playbook_data["id"],
                playbook_data["title"],
                playbook_data["description"],
                json.dumps(playbook_data["sections"]),
                json.dumps(playbook_data["data_steps"]),
                json.dumps(playbook_data["keywords"]),
                playbook_data.get("example_output", ""),
                json.dumps(playbook_data.get("intent_patterns", [])),
                created,
                now,
            ),
        )
    conn.commit()
