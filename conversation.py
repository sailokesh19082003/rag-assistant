import logging
from typing import List, Dict
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

MAX_HISTORY_PAIRS = int(__import__("os").getenv("MAX_HISTORY_PAIRS", "5"))

# In-memory session store: sessionId → deque of message dicts
_sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_PAIRS * 2))


def add_message(session_id: str, role: str, content: str):
    """Append a message to the session history."""
    _sessions[session_id].append({"role": role, "content": content})


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return the conversation history for a session."""
    return list(_sessions[session_id])


def format_history_for_prompt(session_id: str) -> str:
    """Format conversation history as a human-readable string for the prompt."""
    history = get_history(session_id)
    if not history:
        return ""
    lines = []
    for msg in history:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines)


def clear_session(session_id: str):
    """Clear all history for a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Cleared session: {session_id}")
