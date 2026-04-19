"""Memory system — LCM (Lossless Context Management) backend.

Replaces the flat MEMORY.md/HISTORY.md approach with a SQLite-backed
hierarchical summarization DAG. Messages are archived into lcm.db,
then compacted into leaf summaries and condensed up the tree.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


# ---------------------------------------------------------------------------
# Schema — auto-created if lcm.db doesn't exist
# ---------------------------------------------------------------------------

_LCM_SCHEMA = """\
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    seq             INTEGER NOT NULL,
    role            TEXT    NOT NULL CHECK (role IN ('user','assistant','system','tool')),
    content         TEXT    NOT NULL,
    token_count     INTEGER NOT NULL DEFAULT 0,
    tools_used      TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    UNIQUE(conversation_id, seq)
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content=messages, content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS summaries (
    id               TEXT    PRIMARY KEY,
    conversation_id  INTEGER REFERENCES conversations(id),
    depth            INTEGER NOT NULL DEFAULT 0,
    kind             TEXT    NOT NULL CHECK (kind IN ('leaf','condensed')),
    content          TEXT    NOT NULL,
    token_count      INTEGER NOT NULL DEFAULT 0,
    earliest_at      TEXT    NOT NULL,
    latest_at        TEXT    NOT NULL,
    descendant_count INTEGER NOT NULL DEFAULT 0,
    superseded       INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
    content, content=summaries, content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON summaries BEGIN
    INSERT INTO summaries_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS summary_messages (
    summary_id TEXT    NOT NULL REFERENCES summaries(id),
    message_id INTEGER NOT NULL REFERENCES messages(id),
    PRIMARY KEY (summary_id, message_id)
);

CREATE TABLE IF NOT EXISTS summary_parents (
    summary_id TEXT NOT NULL REFERENCES summaries(id),
    parent_id  TEXT NOT NULL REFERENCES summaries(id),
    PRIMARY KEY (summary_id, parent_id)
);

CREATE TABLE IF NOT EXISTS context_items (
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    ordinal         INTEGER NOT NULL,
    item_type       TEXT    NOT NULL CHECK (item_type IN ('message','summary')),
    message_id      INTEGER REFERENCES messages(id),
    summary_id      TEXT    REFERENCES summaries(id),
    PRIMARY KEY (conversation_id, ordinal)
);

CREATE TABLE IF NOT EXISTS large_files (
    id              TEXT    PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    file_name       TEXT,
    mime_type       TEXT,
    byte_size       INTEGER,
    token_count     INTEGER,
    exploration     TEXT,
    storage_path    TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);

CREATE TABLE IF NOT EXISTS lcm_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO lcm_config (key, value) VALUES
    ('leaf_min_fanout',        '8'),
    ('condensed_min_fanout',   '4'),
    ('leaf_target_tokens',     '1200'),
    ('condensed_target_tokens','2000'),
    ('next_leaf_id',           '1'),
    ('next_condensed_id',      '1');

CREATE VIEW IF NOT EXISTS active_summaries AS
SELECT id, depth, kind, token_count, earliest_at, latest_at, descendant_count, created_at
FROM summaries WHERE superseded = 0
ORDER BY depth DESC, earliest_at ASC;

CREATE VIEW IF NOT EXISTS dag_overview AS
SELECT depth, COUNT(*) AS total,
       SUM(CASE WHEN superseded = 0 THEN 1 ELSE 0 END) AS active,
       SUM(CASE WHEN superseded = 1 THEN 1 ELSE 0 END) AS superseded,
       SUM(token_count) AS total_tokens
FROM summaries GROUP BY depth ORDER BY depth;

CREATE VIEW IF NOT EXISTS conversation_stats AS
SELECT c.id, c.session_key, COUNT(m.id) AS message_count,
       SUM(m.token_count) AS total_tokens,
       MIN(m.created_at) AS first_message, MAX(m.created_at) AS last_message
FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id GROUP BY c.id;
"""


# ---------------------------------------------------------------------------
# LLM tool definition for summarization
# ---------------------------------------------------------------------------

_SUMMARIZE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_summary",
            "description": "Save a summary of conversation messages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A concise summary preserving key decisions, outcomes, "
                        "entities, errors, timestamps, and tool usage. ~1200 tokens for leaf, "
                        "~2000 tokens for condensed.",
                    },
                },
                "required": ["summary"],
            },
        },
    }
]

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# LCMStore — replaces MemoryStore
# ---------------------------------------------------------------------------

class LCMStore:
    """LCM-backed memory: SQLite hierarchical summarization DAG with FTS5."""

    LEAF_MIN_FANOUT = 8
    CONDENSATION_MIN = 4
    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.lcm_dir = ensure_dir(workspace / "lcm")
        self.db_path = self.lcm_dir / "lcm.db"
        self._consecutive_failures = 0
        self._init_db()

    def _init_db(self) -> None:
        """Ensure lcm.db exists with full schema."""
        db = sqlite3.connect(str(self.db_path))
        db.executescript(_LCM_SCHEMA)
        db.close()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.db_path))
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def _get_or_create_conversation(self, db: sqlite3.Connection, session_key: str) -> int:
        """Get conversation ID, creating if needed."""
        db.execute(
            "INSERT OR IGNORE INTO conversations (session_key) VALUES (?)",
            (session_key,),
        )
        row = db.execute(
            "SELECT id FROM conversations WHERE session_key = ?", (session_key,)
        ).fetchone()
        return row[0]

    def _get_next_id(self, db: sqlite3.Connection, key: str) -> int:
        """Get and increment a counter from lcm_config."""
        row = db.execute("SELECT value FROM lcm_config WHERE key = ?", (key,)).fetchone()
        val = int(row[0]) if row else 1
        db.execute(
            "UPDATE lcm_config SET value = ? WHERE key = ?", (str(val + 1), key)
        )
        return val

    def get_memory_context(self, session_key: str | None = None) -> str:
        """Build memory context from active (non-superseded) summaries.

        Returns summaries from all conversations (or the specific session_key if
        provided), ordered depth-descending for hierarchical context. This gives
        the LLM a rich view of past interactions beyond recent session messages.
        
        If session_key is provided, loads summaries from that conversation's
        context plus high-level summaries from other conversations.
        """
        db = self._connect()
        try:
            # Load all summaries, not just a budget-limited slice.
            # The LLM needs to see what's available so it can ask to expand
            # specific topics via FTS if needed.
            if session_key:
                # Specific session: load its summaries + global context
                rows = db.execute(
                    "SELECT s.id, s.depth, s.kind, s.content, s.token_count, "
                    "s.earliest_at, s.latest_at, s.conversation_id "
                    "FROM summaries s "
                    "JOIN conversations c ON c.id = s.conversation_id "
                    "WHERE s.superseded = 0 "
                    "AND (c.session_key = ? OR s.depth >= 1) "
                    "ORDER BY s.depth DESC, s.earliest_at ASC",
                    (session_key,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id, depth, kind, content, token_count, "
                    "earliest_at, latest_at, conversation_id "
                    "FROM summaries WHERE superseded = 0 "
                    "ORDER BY depth DESC, earliest_at ASC"
                ).fetchall()
            
            if not rows:
                return ""

            parts = []
            total_tokens = 0
            # Generous budget — summaries are pre-compressed summaries of
            # already-archived messages. Better to include more than to miss
            # relevant context and appear amnesiac.
            max_context_tokens = 12000

            for row in rows:
                sid, depth, kind, content, tokens, earliest, latest, conv_id = row
                if total_tokens + tokens > max_context_tokens:
                    break
                label = f"[{kind} d{depth}] {earliest[:10]}..{latest[:10]}"
                parts.append(f"### {label}\n{content}")
                total_tokens += tokens

            if not parts:
                return ""

            return "## Long-term Memory (LCM)\n\n" + "\n\n".join(parts)
        finally:
            db.close()

    # ----- Message archival -----

    def _archive_messages(
        self, db: sqlite3.Connection, conv_id: int, messages: list[dict]
    ) -> list[int]:
        """Insert messages into lcm.db. Returns list of new message IDs."""
        row = db.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()
        seq = row[0]
        message_ids = []

        for msg in messages:
            content = msg.get("content", "")
            if not content or not str(content).strip():
                continue

            role = msg.get("role", "user")
            if role not in ("user", "assistant", "system", "tool"):
                continue

            content_str = str(content)
            timestamp = msg.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))

            tools_used = None
            if msg.get("tool_calls"):
                tool_names = [
                    tc.get("function", {}).get("name", "")
                    for tc in msg["tool_calls"]
                    if isinstance(tc, dict) and tc.get("function")
                ]
                if tool_names:
                    tools_used = ",".join(tool_names)

            seq += 1
            token_count = _estimate_tokens(content_str)

            try:
                db.execute(
                    "INSERT INTO messages (conversation_id, seq, role, content, token_count, tools_used, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (conv_id, seq, role, content_str, token_count, tools_used, timestamp),
                )
                msg_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
                message_ids.append(msg_id)
            except sqlite3.IntegrityError:
                # Duplicate seq — message already archived (e.g., by cron)
                continue

        db.execute(
            "UPDATE conversations SET updated_at = strftime('%Y-%m-%dT%H:%M:%S','now') WHERE id = ?",
            (conv_id,),
        )
        return message_ids

    # ----- Leaf compaction -----

    def _get_uncompacted_messages(
        self, db: sqlite3.Connection, conv_id: int, limit: int
    ) -> list[tuple]:
        """Get oldest messages not yet covered by any summary."""
        return db.execute(
            "SELECT m.id, m.seq, m.role, m.content, m.created_at "
            "FROM messages m "
            "WHERE m.conversation_id = ? "
            "AND m.id NOT IN (SELECT message_id FROM summary_messages) "
            "ORDER BY m.seq ASC LIMIT ?",
            (conv_id, limit),
        ).fetchall()

    def _count_uncompacted(self, db: sqlite3.Connection, conv_id: int) -> int:
        return db.execute(
            "SELECT COUNT(*) FROM messages m "
            "WHERE m.conversation_id = ? "
            "AND m.id NOT IN (SELECT message_id FROM summary_messages)",
            (conv_id,),
        ).fetchone()[0]

    def _create_leaf_summary(
        self,
        db: sqlite3.Connection,
        conv_id: int,
        message_rows: list[tuple],
        summary_text: str,
    ) -> str:
        """Insert a leaf summary and link it to source messages."""
        leaf_id_num = self._get_next_id(db, "next_leaf_id")
        leaf_id = f"leaf_{leaf_id_num:03d}"

        earliest = message_rows[0][4]  # created_at of first
        latest = message_rows[-1][4]  # created_at of last
        token_count = _estimate_tokens(summary_text)

        db.execute(
            "INSERT INTO summaries (id, conversation_id, depth, kind, content, token_count, earliest_at, latest_at) "
            "VALUES (?, ?, 0, 'leaf', ?, ?, ?, ?)",
            (leaf_id, conv_id, summary_text, token_count, earliest, latest),
        )

        for row in message_rows:
            db.execute(
                "INSERT OR IGNORE INTO summary_messages (summary_id, message_id) VALUES (?, ?)",
                (leaf_id, row[0]),
            )

        logger.info("LCM leaf compaction: {} covering {} messages", leaf_id, len(message_rows))
        return leaf_id

    # ----- Condensation -----

    def _get_condensation_candidates(self, db: sqlite3.Connection, conv_id: int) -> tuple[int, list] | None:
        """Find the lowest depth with >= CONDENSATION_MIN active summaries."""
        row = db.execute(
            "SELECT depth, COUNT(*) AS cnt FROM summaries "
            "WHERE superseded = 0 AND conversation_id = ? "
            "GROUP BY depth HAVING cnt >= ? ORDER BY depth LIMIT 1",
            (conv_id, self.CONDENSATION_MIN),
        ).fetchone()
        if not row:
            return None

        depth = row[0]
        summaries = db.execute(
            "SELECT id, content, token_count, earliest_at, latest_at FROM summaries "
            "WHERE superseded = 0 AND depth = ? AND conversation_id = ? "
            "ORDER BY earliest_at ASC LIMIT ?",
            (depth, conv_id, self.CONDENSATION_MIN),
        ).fetchall()
        return depth, summaries

    def _create_condensed_summary(
        self,
        db: sqlite3.Connection,
        conv_id: int,
        depth: int,
        parent_rows: list[tuple],
        summary_text: str,
    ) -> str:
        """Insert a condensed summary and supersede its parents."""
        cond_id_num = self._get_next_id(db, "next_condensed_id")
        cond_id = f"condensed_{cond_id_num:03d}"

        new_depth = depth + 1
        earliest = parent_rows[0][3]
        latest = parent_rows[-1][4]
        token_count = _estimate_tokens(summary_text)

        db.execute(
            "INSERT INTO summaries (id, conversation_id, depth, kind, content, token_count, "
            "earliest_at, latest_at, descendant_count) VALUES (?, ?, ?, 'condensed', ?, ?, ?, ?, ?)",
            (cond_id, conv_id, new_depth, summary_text, token_count, earliest, latest,
             len(parent_rows)),
        )

        for row in parent_rows:
            db.execute(
                "INSERT INTO summary_parents (summary_id, parent_id) VALUES (?, ?)",
                (cond_id, row[0]),
            )

        parent_ids = [row[0] for row in parent_rows]
        placeholders = ",".join("?" * len(parent_ids))
        db.execute(
            f"UPDATE summaries SET superseded = 1 WHERE id IN ({placeholders})",
            parent_ids,
        )

        logger.info("LCM condensation: {} (depth {}) from {} parents", cond_id, new_depth, len(parent_rows))
        return cond_id

    # ----- LLM summarization -----

    async def _llm_summarize(
        self,
        provider: LLMProvider,
        model: str,
        text_to_summarize: str,
        kind: str,
    ) -> str | None:
        """Ask the LLM to produce a summary via tool call."""
        target = "~1200 tokens" if kind == "leaf" else "~2000 tokens"
        chat_messages = [
            {
                "role": "system",
                "content": f"You are a memory consolidation agent. Summarize the conversation "
                f"into {target}. Preserve timestamps, decisions, outcomes, entities, "
                f"errors, and tool usage. Call the save_summary tool with your summary.",
            },
            {"role": "user", "content": text_to_summarize},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_summary"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SUMMARIZE_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SUMMARIZE_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                # Fallback: use content directly if available
                if response.content and len(response.content.strip()) > 50:
                    return response.content.strip()
                logger.warning("LCM summarization: LLM did not call save_summary")
                return None

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(args, list):
                args = args[0] if args else {}
            if isinstance(args, dict):
                summary = args.get("summary", "")
                if summary and len(str(summary).strip()) > 20:
                    return str(summary).strip()

            logger.warning("LCM summarization: empty or invalid summary from LLM")
            return None
        except Exception:
            logger.exception("LCM summarization failed")
            return None

    @staticmethod
    def _format_messages_for_summary(messages: list[dict] | list[tuple]) -> str:
        """Format messages for LLM summarization prompt."""
        lines = []
        for msg in messages:
            if isinstance(msg, tuple):
                # (id, seq, role, content, created_at)
                _, seq, role, content, ts = msg
                lines.append(f"[{ts[:16]}] {role.upper()}: {content}")
            else:
                content = msg.get("content", "")
                if not content:
                    continue
                ts = msg.get("timestamp", "?")[:16]
                role = msg.get("role", "?").upper()
                tools = ""
                if msg.get("tool_calls"):
                    tool_names = [
                        tc.get("function", {}).get("name", "")
                        for tc in msg["tool_calls"]
                        if isinstance(tc, dict) and tc.get("function")
                    ]
                    if tool_names:
                        tools = f" [tools: {', '.join(tool_names)}]"
                lines.append(f"[{ts}] {role}{tools}: {content}")
        return "\n".join(lines)

    # ----- Main consolidation entry point -----

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        session_key: str = "",
    ) -> bool:
        """Archive messages to LCM and perform leaf compaction + condensation.

        Called by MemoryConsolidator when context exceeds budget.
        """
        if not messages:
            return True

        db = self._connect()
        try:
            conv_id = self._get_or_create_conversation(db, session_key or "default")

            # Step 1: Archive messages
            self._archive_messages(db, conv_id, messages)
            db.commit()

            # Step 2: Leaf compaction if needed
            uncompacted_count = self._count_uncompacted(db, conv_id)
            if uncompacted_count >= self.LEAF_MIN_FANOUT:
                msg_rows = self._get_uncompacted_messages(db, conv_id, self.LEAF_MIN_FANOUT)
                if msg_rows:
                    text = self._format_messages_for_summary(msg_rows)
                    summary = await self._llm_summarize(provider, model, text, "leaf")
                    if summary:
                        self._create_leaf_summary(db, conv_id, msg_rows, summary)
                        db.commit()
                    else:
                        # Fallback: truncated raw text
                        fallback = text[:2048] + "\n[Truncated for context management]"
                        self._create_leaf_summary(db, conv_id, msg_rows, fallback)
                        db.commit()

            # Step 3: Condensation if needed
            cand = self._get_condensation_candidates(db, conv_id)
            if cand:
                depth, parent_rows = cand
                parent_texts = "\n\n---\n\n".join(
                    f"[Summary {r[0]}, {r[3][:10]}..{r[4][:10]}]\n{r[1]}" for r in parent_rows
                )
                summary = await self._llm_summarize(provider, model, parent_texts, "condensed")
                if summary:
                    self._create_condensed_summary(db, conv_id, depth, parent_rows, summary)
                    db.commit()

            self._consecutive_failures = 0
            logger.info("LCM consolidation done: {} messages archived", len(messages))
            return True
        except Exception:
            logger.exception("LCM consolidation failed")
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
                # Emergency fallback: just archive messages without summarization
                try:
                    db2 = self._connect()
                    conv_id = self._get_or_create_conversation(db2, session_key or "default")
                    self._archive_messages(db2, conv_id, messages)
                    db2.commit()
                    db2.close()
                    self._consecutive_failures = 0
                    logger.warning("LCM fallback: raw-archived {} messages", len(messages))
                    return True
                except Exception:
                    logger.exception("LCM fallback archival also failed")
            return False
        finally:
            db.close()


# Keep MemoryStore as alias for backward compatibility
MemoryStore = LCMStore


# ---------------------------------------------------------------------------
# MemoryConsolidator — orchestrates when to consolidate
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5
    _SAFETY_BUFFER = 1024

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ):
        self.store = LCMStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(
        self, messages: list[dict[str, object]], session_key: str = ""
    ) -> bool:
        """Archive a selected message chunk into LCM."""
        return await self.store.consolidate(messages, self.provider, self.model, session_key)

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(self, messages: list[dict[str, object]], session_key: str = "") -> bool:
        """Archive messages with guaranteed persistence."""
        if not messages:
            return True
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self.consolidate_messages(messages, session_key):
                return True
        return True

    async def maybe_consolidate_by_tokens(self, session: Session, session_summary: str | None = None) -> None:
        """Loop: archive old messages until prompt fits within safe budget."""
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < budget:
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk, session.key):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
