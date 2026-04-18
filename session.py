import uuid
import json
import os
import redis
from dotenv import load_dotenv

load_dotenv()

SESSION_TTL = 60 * 60 * 24  # 24 hours
MAX_HISTORY = 6

# Single Redis client reused across all requests
_client: redis.Redis = None


def get_redis() -> redis.Redis:
    """Return a shared Redis client, creating it on first call."""
    global _client
    if _client is None:
        url = os.getenv("REDIS_URL")
        if not url:
            raise RuntimeError("REDIS_URL is not set in environment variables.")
        _client = redis.Redis.from_url(url, decode_responses=True)
    return _client


def create_session_id() -> str:
    """Generate a new session ID and initialise an empty history in Redis."""
    session_id = str(uuid.uuid4())
    r = get_redis()
    r.setex(f"history:{session_id}", SESSION_TTL, json.dumps([]))
    return session_id


def get_history(session_id: str) -> list[tuple[str, str]]:
    """Fetch conversation history for a session. Returns [] if not found."""
    r = get_redis()
    data = r.get(f"history:{session_id}")
    if not data:
        return []
    return [tuple(pair) for pair in json.loads(data)]


def save_history(session_id: str, history: list[tuple[str, str]]):
    """Persist conversation history, keeping only the last MAX_HISTORY turns."""
    r = get_redis()
    trimmed = history[-MAX_HISTORY:]
    r.setex(f"history:{session_id}", SESSION_TTL, json.dumps(trimmed))


def delete_history(session_id: str):
    """Delete a session's history from Redis."""
    r = get_redis()
    r.delete(f"history:{session_id}")


def session_exists(session_id: str) -> bool:
    """Return True if the session key exists (and hasn't expired) in Redis."""
    r = get_redis()
    return r.exists(f"history:{session_id}") == 1
