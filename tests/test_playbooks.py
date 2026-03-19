"""Tests for playbook data model, CRUD, and seeding."""

import tempfile
from pathlib import Path

import pytest

try:
    import sqlite_vec
    from fastembed import TextEmbedding
    HAS_INSIGHTS = True
except ImportError:
    HAS_INSIGHTS = False


@pytest.mark.skipif(not HAS_INSIGHTS, reason="insights deps not installed")
class TestPlaybooks:
    """Test playbook CRUD operations."""

    def _setup_db(self, tmp_path):
        from claude_session_commons.insights import get_db, init_db
        db_path = str(tmp_path / "test.db")
        conn = get_db(db_path)
        init_db(conn)
        return conn

    def test_seed_defaults(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import list_playbooks
            pbs = list_playbooks(conn)
            assert len(pbs) >= 2
            ids = [pb.id for pb in pbs]
            assert "executive-brief" in ids
            assert "project-status" in ids
        finally:
            conn.close()

    def test_executive_brief_sections(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import get_playbook
            pb = get_playbook(conn, "executive-brief")
            assert pb is not None
            assert pb.title == "Executive Brief"
            assert len(pb.sections) == 10
            section_names = [s["name"] for s in pb.sections]
            assert "Executive Summary" in section_names
            assert "Where We Were" in section_names
            assert "Blockers" in section_names
            assert "Risks & Mitigations" in section_names
        finally:
            conn.close()

    def test_create_playbook(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import create_playbook, get_playbook
            pb = create_playbook(
                conn,
                title="Weekly Standup",
                description="Template for weekly standup notes",
                sections=[
                    {"name": "What I Did", "description": "Work completed", "prompt_hint": "List completed items"},
                    {"name": "What's Next", "description": "Upcoming work", "prompt_hint": "List planned items"},
                ],
                keywords=["standup", "weekly"],
            )
            assert pb is not None
            assert pb.title == "Weekly Standup"
            assert len(pb.sections) == 2
            assert pb.keywords == ["standup", "weekly"]

            # Verify it persists
            fetched = get_playbook(conn, pb.id)
            assert fetched is not None
            assert fetched.title == "Weekly Standup"
        finally:
            conn.close()

    def test_update_playbook(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import create_playbook, update_playbook
            pb = create_playbook(
                conn, title="Original", description="Desc",
                sections=[{"name": "S1", "description": "D1", "prompt_hint": "H1"}],
            )

            updated = update_playbook(conn, pb.id, title="Updated Title")
            assert updated is not None
            assert updated.title == "Updated Title"
            assert updated.description == "Desc"  # Unchanged
            assert len(updated.sections) == 1  # Unchanged
        finally:
            conn.close()

    def test_update_sections(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import create_playbook, update_playbook
            pb = create_playbook(
                conn, title="Test", description="Desc",
                sections=[{"name": "S1", "description": "D1", "prompt_hint": "H1"}],
            )

            new_sections = [
                {"name": "S1", "description": "D1", "prompt_hint": "H1"},
                {"name": "S2", "description": "D2", "prompt_hint": "H2"},
            ]
            updated = update_playbook(conn, pb.id, sections=new_sections)
            assert len(updated.sections) == 2
        finally:
            conn.close()

    def test_delete_playbook(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import create_playbook, delete_playbook, get_playbook
            pb = create_playbook(
                conn, title="To Delete", description="Will be deleted",
                sections=[{"name": "S1", "description": "D1", "prompt_hint": "H1"}],
            )

            assert delete_playbook(conn, pb.id) is True
            assert get_playbook(conn, pb.id) is None
        finally:
            conn.close()

    def test_delete_nonexistent(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import delete_playbook
            assert delete_playbook(conn, "nope") is False
        finally:
            conn.close()

    def test_search_playbooks(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import search_playbooks
            results = search_playbooks(conn, "executive")
            assert len(results) >= 1
            assert results[0].id == "executive-brief"
        finally:
            conn.close()

    def test_search_by_keyword(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import search_playbooks
            results = search_playbooks(conn, "leadership")
            assert len(results) >= 1
        finally:
            conn.close()

    def test_search_no_results(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import search_playbooks
            results = search_playbooks(conn, "zzzznonexistent")
            assert results == []
        finally:
            conn.close()

    def test_list_excludes_deleted(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import create_playbook, delete_playbook, list_playbooks
            pb = create_playbook(
                conn, title="Temp", description="Temp",
                sections=[{"name": "S1", "description": "D1", "prompt_hint": "H1"}],
            )
            count_before = len(list_playbooks(conn))
            delete_playbook(conn, pb.id)
            count_after = len(list_playbooks(conn))
            assert count_after == count_before - 1
        finally:
            conn.close()

    def test_seed_idempotent(self, tmp_path):
        """Seeding twice should not create duplicate playbooks."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import seed_defaults, list_playbooks
            # init_db already called seed_defaults once
            count1 = len(list_playbooks(conn))
            seed_defaults(conn)
            count2 = len(list_playbooks(conn))
            assert count1 == count2
        finally:
            conn.close()

    def test_example_output_stored(self, tmp_path):
        """Executive brief should have a non-empty example_output."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import get_playbook
            pb = get_playbook(conn, "executive-brief")
            assert pb is not None
            assert len(pb.example_output) > 500  # Should be a substantial example
            assert "Executive Summary" in pb.example_output
            assert "Where We Were" in pb.example_output
            assert "Timeline" in pb.example_output
        finally:
            conn.close()

    def test_update_nonexistent(self, tmp_path):
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import update_playbook
            result = update_playbook(conn, "nope", title="New")
            assert result is None
        finally:
            conn.close()

    # ── Project Status playbook tests ─────────────────────────

    def test_project_status_sections(self, tmp_path):
        """Project status playbook should have 5 sections."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import get_playbook
            pb = get_playbook(conn, "project-status")
            assert pb is not None
            assert pb.title == "Project Status Check"
            assert len(pb.sections) == 5
            section_names = [s["name"] for s in pb.sections]
            assert "Current State" in section_names
            assert "Recent Activity" in section_names
            assert "What's Next" in section_names
            assert "Blockers & Concerns" in section_names
        finally:
            conn.close()

    def test_project_status_example_output(self, tmp_path):
        """Project status playbook should have a non-empty example_output."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import get_playbook
            pb = get_playbook(conn, "project-status")
            assert pb is not None
            assert len(pb.example_output) > 200
            assert "Current State" in pb.example_output
            assert "Recent Activity" in pb.example_output
        finally:
            conn.close()

    # ── Intent pattern matching tests ─────────────────────────

    def test_match_status_question(self, tmp_path):
        """'where are we on cerebro?' should match project-status."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import match_playbook_by_intent
            pb = match_playbook_by_intent(conn, "where are we on cerebro?")
            assert pb is not None
            assert pb.id == "project-status"
        finally:
            conn.close()

    def test_match_executive_brief(self, tmp_path):
        """'write an executive brief' should match executive-brief."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import match_playbook_by_intent
            pb = match_playbook_by_intent(conn, "write an executive brief on greenmark")
            assert pb is not None
            assert pb.id == "executive-brief"
        finally:
            conn.close()

    def test_match_leave_off(self, tmp_path):
        """'where did we leave off with data-daemon?' should match project-status."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import match_playbook_by_intent
            pb = match_playbook_by_intent(conn, "where did we leave off with data-daemon?")
            assert pb is not None
            assert pb.id == "project-status"
        finally:
            conn.close()

    def test_match_catch_up(self, tmp_path):
        """'catch me up on weekly updates' should match project-status."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import match_playbook_by_intent
            pb = match_playbook_by_intent(conn, "catch me up on weekly updates")
            assert pb is not None
            assert pb.id == "project-status"
        finally:
            conn.close()

    def test_match_how_going(self, tmp_path):
        """'how is greenmark going?' should match project-status."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import match_playbook_by_intent
            pb = match_playbook_by_intent(conn, "how is greenmark going?")
            assert pb is not None
            assert pb.id == "project-status"
        finally:
            conn.close()

    def test_no_match(self, tmp_path):
        """'hello' should not match any playbook."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import match_playbook_by_intent
            pb = match_playbook_by_intent(conn, "hello")
            assert pb is None
        finally:
            conn.close()

    def test_malformed_pattern(self, tmp_path):
        """Bad regex in intent_patterns should not crash."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import create_playbook, match_playbook_by_intent
            create_playbook(
                conn, title="Bad Regex", description="Has bad pattern",
                sections=[{"name": "S1", "description": "D1", "prompt_hint": "H1"}],
                intent_patterns=["(?i)valid", "[invalid("],
            )
            # Should not crash — skips bad patterns, matches good ones
            pb = match_playbook_by_intent(conn, "this is valid content")
            assert pb is not None
            assert pb.title == "Bad Regex"

            # Non-matching message should return None, not crash
            pb2 = match_playbook_by_intent(conn, "nothing matches here zzz")
            assert pb2 is None
        finally:
            conn.close()

    def test_create_with_intent_patterns(self, tmp_path):
        """Creating a playbook with intent_patterns should persist them."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import create_playbook, get_playbook
            pb = create_playbook(
                conn, title="Custom", description="Test patterns",
                sections=[{"name": "S1", "description": "D1", "prompt_hint": "H1"}],
                intent_patterns=[r"(?i)custom\s+query"],
            )
            fetched = get_playbook(conn, pb.id)
            assert fetched.intent_patterns == [r"(?i)custom\s+query"]
        finally:
            conn.close()

    def test_match_playbook_combined(self, tmp_path):
        """match_playbook() should try intent patterns first, then semantic."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import match_playbook
            # Intent pattern match (fast path)
            pb = match_playbook(conn, "where are we on cerebro?", model=None)
            assert pb is not None
            assert pb.id == "project-status"

            # No match without model fallback
            pb2 = match_playbook(conn, "hello there", model=None)
            assert pb2 is None
        finally:
            conn.close()

    def test_seed_includes_intent_patterns(self, tmp_path):
        """Both seed playbooks should have intent_patterns set."""
        conn = self._setup_db(tmp_path)
        try:
            from claude_session_commons.playbooks import get_playbook
            eb = get_playbook(conn, "executive-brief")
            ps = get_playbook(conn, "project-status")
            assert len(eb.intent_patterns) >= 2
            assert len(ps.intent_patterns) >= 5
        finally:
            conn.close()
