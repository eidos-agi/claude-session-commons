"""Full-screen chat interface for conversational session discovery.

Textual Screen — minimal chrome, Claude Code feel.
Press C from session list to open, Esc to return.
"""

import re

from textual import events, work
from textual.app import ComposeResult
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Input, RichLog, Static

from ..chat_agent import ChatAgent
from ..paths import shorten_path


def _esc(text: str) -> str:
    """Escape Rich markup in user/agent text."""
    return str(text).replace("[", "\\[")


class ChatScreen(Screen):
    """Modal chat for conversational session discovery."""

    ALLOW_SELECT = True

    DEFAULT_CSS = """
    ChatScreen {
        layout: vertical;
        background: $surface;
    }
    ChatScreen #chat-log {
        height: 1fr;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    ChatScreen #chat-picker {
        height: auto;
        max-height: 8;
        padding: 0 2;
        border-top: solid $primary;
        display: none;
    }
    ChatScreen #chat-picker.visible {
        display: block;
    }
    ChatScreen #chat-status {
        height: 1;
        padding: 0 2;
    }
    ChatScreen #chat-input {
        dock: bottom;
        margin: 0 1;
        height: 3;
    }
    """

    BINDINGS = [
        ("escape", "dismiss", "Back to sessions"),
    ]

    class _ResponseReady(Message):
        """Internal: LLM response arrived from background thread."""
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, agent: ChatAgent, skip_permissions: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._agent = agent
        self._skip_permissions = skip_permissions
        self._thinking = False
        # Pick mode: navigable session results
        self._pick_refs: list[int] = []  # #N numbers from last response
        self._pick_idx: int = 0

    def compose(self) -> ComposeResult:
        yield RichLog(id="chat-log", wrap=True, markup=True)
        yield Static("", id="chat-picker")
        yield Static("", id="chat-status")
        yield Input(placeholder="Ask about your sessions... (Esc to go back)", id="chat-input")

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        n_sessions, n_projects = self._agent.welcome_info()
        n_chats = len(self._agent._past_chats)
        info = f"{n_sessions} sessions across {n_projects} projects"
        if n_chats:
            info += f" · {n_chats} past chats"
        log.write(f"[bold]Session navigator[/] [dim]· {info}[/]")
        log.write("[dim]Ask anything. Arrow through results, Enter to open. Esc to go back.[/]")
        log.write("")
        self.query_one("#chat-input", Input).focus()

    # ── Input handling ────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat-input":
            return
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self._clear_pick_mode()

        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold cyan]> {_esc(text)}[/]")

        # Check for direct resume command (no LLM needed)
        num = self._agent.parse_resume_request(text)
        if num is not None:
            self._handle_resume(num)
            return

        self._set_status("[dim italic]thinking...[/]")
        self._thinking = True
        self._get_response(text)

    def on_key(self, event: events.Key) -> None:
        """Handle arrow keys for pick mode navigation."""
        if not self._pick_refs:
            return

        if event.key == "up":
            self._pick_idx = max(0, self._pick_idx - 1)
            self._render_pick_bar()
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            self._pick_idx = min(len(self._pick_refs) - 1, self._pick_idx + 1)
            self._render_pick_bar()
            event.prevent_default()
            event.stop()
        elif event.key == "enter" and self.query_one("#chat-input", Input).value == "":
            # Enter on empty input = open selected result
            num = self._pick_refs[self._pick_idx]
            self._handle_resume(num)
            event.prevent_default()
            event.stop()

    # ── LLM response ──────────────────────────────────────

    @work(thread=True)
    def _get_response(self, text: str) -> None:
        """Call the agent in a background thread."""
        response = self._agent.respond(text)
        self.post_message(self._ResponseReady(response))

    def on_chat_screen__response_ready(self, message: _ResponseReady) -> None:
        if not self.is_attached:
            return
        log = self.query_one("#chat-log", RichLog)
        log.write("")
        for line in message.text.split("\n"):
            log.write(_esc(line))
        log.write("")
        self._set_status("")
        self._thinking = False

        # Extract #N references and enter pick mode if any found
        refs = self._extract_refs(message.text)
        if refs:
            self._enter_pick_mode(refs)

        self.query_one("#chat-input", Input).focus()

    # ── Pick mode (navigable results) ─────────────────────

    def _extract_refs(self, text: str) -> list[int]:
        """Find all #N session references in a response."""
        nums = []
        seen = set()
        for m in re.finditer(r"#(\d+)", text):
            n = int(m.group(1))
            if n not in seen and self._agent.get_ref_by_num(n) is not None:
                nums.append(n)
                seen.add(n)
        return nums

    def _enter_pick_mode(self, refs: list[int]) -> None:
        self._pick_refs = refs
        self._pick_idx = 0
        self.query_one("#chat-picker").add_class("visible")
        self._render_pick_bar()

    def _clear_pick_mode(self) -> None:
        self._pick_refs = []
        self._pick_idx = 0
        self.query_one("#chat-picker").remove_class("visible")

    def _render_pick_bar(self) -> None:
        """Render the pick bar showing navigable results."""
        parts: list[str] = []
        for i, num in enumerate(self._pick_refs):
            ref = self._agent.get_ref_by_num(num)
            if ref is None:
                continue
            short = shorten_path(ref.project_dir)
            title = ref.title[:40]
            if i == self._pick_idx:
                parts.append(f"[bold reverse] #{num} {title} [/] [dim]{short} · {ref.age}[/]")
            else:
                parts.append(f"  [dim]#{num}[/] {title}  [dim]{short} · {ref.age}[/]")
        picker = self.query_one("#chat-picker", Static)
        hint = "[dim]  ↑↓ navigate · Enter to open · type to keep chatting[/]"
        picker.update("\n".join(parts) + "\n" + hint)

    # ── Resume ────────────────────────────────────────────

    def _handle_resume(self, num: int) -> None:
        session = self._agent.get_session_by_num(num)
        if session is None:
            log = self.query_one("#chat-log", RichLog)
            n = len(self._agent._catalog)
            log.write(f"[red]No session #{num} found. I have #1 through #{n}.[/]")
            return

        self._agent.log_resume(num)
        cmd = self._agent.build_resume_cmd(num, skip_permissions=self._skip_permissions)
        if not cmd:
            return

        ref = self._agent.get_ref_by_num(num)
        from .session_picker import SessionPickerPanel
        self.app.post_message(
            SessionPickerPanel.SessionSelected("resume", ref.original_idx, cmd=cmd)
        )

    # ── Helpers ───────────────────────────────────────────

    def _set_status(self, text: str) -> None:
        self.query_one("#chat-status", Static).update(text)

    def action_dismiss(self) -> None:
        self._agent.log_end()
        self.app.pop_screen()
