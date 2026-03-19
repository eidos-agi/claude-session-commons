"""In-memory conversation state for RAG chat sessions.

Simple state manager for single-user local dashboard.
Each chat session has a message history and an async event queue
for streaming SSE events from the agent to the browser.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ChatSession:
    """A single chat conversation."""

    chat_id: str
    created_at: float = field(default_factory=time.time)
    messages: list[dict] = field(default_factory=list)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    active: bool = False


class ChatStateManager:
    """Manages in-memory chat sessions with TTL cleanup."""

    TTL_SECONDS = 3600  # 1 hour
    MAX_SESSIONS = 20

    def __init__(self):
        self._sessions: dict[str, ChatSession] = {}

    def create_session(self) -> str:
        """Create a new chat session. Returns chat_id."""
        self._cleanup_stale()
        chat_id = str(uuid.uuid4())[:8]
        self._sessions[chat_id] = ChatSession(chat_id=chat_id)
        return chat_id

    def get_session(self, chat_id: str) -> ChatSession | None:
        return self._sessions.get(chat_id)

    def add_message(self, chat_id: str, role: str, content: str):
        session = self._sessions.get(chat_id)
        if session:
            session.messages.append({"role": role, "content": content})

    async def push_event(self, chat_id: str, event: dict):
        session = self._sessions.get(chat_id)
        if session:
            await session.event_queue.put(event)

    async def consume_events(self, chat_id: str):
        """Yield SSE events until a 'done' or 'error' event.

        Uses a timeout to avoid blocking forever on idle connections.
        The EventSource client will automatically reconnect.
        """
        session = self._sessions.get(chat_id)
        if not session:
            return
        while True:
            try:
                event = await asyncio.wait_for(session.event_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a keepalive comment to prevent connection timeout
                yield {"type": "keepalive", "html": ""}
                continue
            yield event
            if event.get("type") in ("done", "error"):
                break

    def _cleanup_stale(self):
        now = time.time()
        stale = [
            k for k, v in self._sessions.items()
            if now - v.created_at > self.TTL_SECONDS
        ]
        for k in stale:
            del self._sessions[k]
        # Cap total sessions
        if len(self._sessions) >= self.MAX_SESSIONS:
            oldest = sorted(
                self._sessions.items(), key=lambda x: x[1].created_at
            )
            for k, _ in oldest[: len(self._sessions) - self.MAX_SESSIONS + 1]:
                del self._sessions[k]
