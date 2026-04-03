# LCM — Lossless Context Management

Nanobot's memory system uses a SQLite-backed hierarchical summarization DAG instead of flat markdown files. When context fills up, old messages are archived, summarized into leaf nodes, and condensed upward into a tree — preserving full conversation history with FTS5 search.

## Architecture

```
Live Session (JSONL)          LCM Database (lcm.db)
┌─────────────────┐           ┌──────────────────────────────┐
│ msg 1            │           │ messages (immutable store)    │
│ msg 2            │  ──────>  │ summaries (DAG)              │
│ ...              │  archive  │ summary_messages (links)      │
│ msg N            │           │ summary_parents (tree edges)  │
│                  │           │ messages_fts / summaries_fts  │
│ last_consolidated│           └──────────────────────────────┘
└─────────────────┘                      │
                                         ▼
                               ┌──────────────────┐
                               │ System Prompt     │
                               │ (active summaries │
                               │  loaded as memory)│
                               └──────────────────┘
```

## How It Works

### 1. Archival

When the session's token count exceeds the context budget (`contextWindowTokens - maxCompletionTokens - 1024`), the consolidator picks a chunk of old messages and inserts them into `lcm.db`:

```
Session: [msg1, msg2, ..., msg20, msg21, ..., msg40]
                ▲                        ▲
         last_consolidated          current turn
```

Messages between `last_consolidated` and the chosen boundary are archived to `lcm.db`, then `last_consolidated` advances so they drop out of the LLM context.

### 2. Leaf Compaction (Depth 0)

When **8 or more** archived messages are not yet covered by any summary, the system:

1. Reads the 8 oldest uncompacted messages
2. Asks the LLM to produce a ~1200 token summary preserving decisions, outcomes, timestamps, entities, errors, and tool usage
3. Inserts a `leaf` summary at depth 0 and links it to the source messages

```
msg1 ─┐
msg2 ─┤
msg3 ─┤
msg4 ─┼──► leaf_001 (depth 0, ~1200 tokens)
msg5 ─┤
msg6 ─┤
msg7 ─┤
msg8 ─┘
```

If summarization fails, raw text is truncated to 2048 characters as a fallback.

### 3. Condensation (Depth 1+)

When **4 or more** non-superseded summaries exist at the same depth, the system:

1. Reads the 4 oldest active summaries at that depth
2. Asks the LLM to produce a ~2000 token higher-level summary
3. Inserts a `condensed` summary at depth + 1
4. Marks the parent summaries as `superseded = 1`

```
leaf_001 ─┐
leaf_002 ─┼──► condensed_001 (depth 1, ~2000 tokens)
leaf_003 ─┤
leaf_004 ─┘

leaf_005 ─┐
leaf_006 ─┼──► condensed_002 (depth 1)
leaf_007 ─┤
leaf_008 ─┘

condensed_001 ─┐
condensed_002 ─┼──► condensed_003 (depth 2)
condensed_003 ─┤
condensed_004 ─┘
```

The tree grows upward as conversations get longer. Only non-superseded summaries (the tree tops) are loaded into context.

### 4. Context Loading

On each LLM call, `get_memory_context()` reads active (non-superseded) summaries from `lcm.db`, ordered by depth descending and time ascending. These are injected into the system prompt under `# Memory`, giving the bot a hierarchical view of past conversations.

A token budget (default 4000) limits how much memory context is included, preventing the memory itself from consuming too much of the context window.

### 5. Retrieval

The bot can search past conversations using FTS5 full-text search:

```sql
-- Search messages
SELECT m.id, m.role, snippet(messages_fts, 0, '>>>', '<<<', '...', 40)
FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid
WHERE messages_fts MATCH 'keyword' ORDER BY rank LIMIT 10;

-- Search summaries
SELECT s.id, s.depth, snippet(summaries_fts, 0, '>>>', '<<<', '...', 40)
FROM summaries_fts JOIN summaries s ON s.rowid = summaries_fts.rowid
WHERE summaries_fts MATCH 'keyword' ORDER BY rank LIMIT 10;

-- Expand a summary to its source messages
SELECT m.seq, m.role, m.content
FROM summary_messages sm JOIN messages m ON m.id = sm.message_id
WHERE sm.summary_id = 'leaf_001' ORDER BY m.seq;
```

## Database Schema

### Tables

| Table | Purpose |
|-------|---------|
| `conversations` | One row per session (keyed by `session_key` like `telegram:12345`) |
| `messages` | Immutable message store with FTS5 index |
| `summaries` | Hierarchical summary DAG — `depth`, `kind` (leaf/condensed), `superseded` flag |
| `summary_messages` | Links leaf summaries to their source messages |
| `summary_parents` | Links condensed summaries to their parent summaries |
| `lcm_config` | Counters and thresholds (`next_leaf_id`, `leaf_min_fanout`, etc.) |

### Views

| View | Purpose |
|------|---------|
| `active_summaries` | Non-superseded summaries, highest depth first |
| `dag_overview` | Summary counts by depth (active vs superseded) |
| `conversation_stats` | Message counts and token totals per conversation |

## Configuration

Thresholds are stored in `lcm_config` and can be tuned:

| Key | Default | Description |
|-----|---------|-------------|
| `leaf_min_fanout` | 8 | Minimum uncompacted messages before leaf compaction triggers |
| `condensed_min_fanout` | 4 | Minimum active summaries at a depth before condensation triggers |
| `leaf_target_tokens` | 1200 | Target token count for leaf summaries |
| `condensed_target_tokens` | 2000 | Target token count for condensed summaries |

The context budget for consolidation is derived from `contextWindowTokens` in `config.json`:

```
budget = contextWindowTokens - maxCompletionTokens - 1024 (safety buffer)
target = budget / 2
```

Consolidation triggers when estimated prompt tokens exceed `budget` and keeps archiving until tokens drop below `target`.

## Comparison with Previous System

| | MEMORY.md (old) | LCM (new) |
|---|---|---|
| Storage | Flat markdown files | SQLite with FTS5 |
| Summarization | Single-pass → overwrite | Hierarchical tree |
| History | Append-only log (HISTORY.md) | Immutable message store + searchable summaries |
| Retrieval | `grep` on flat file | FTS5 full-text search across messages and summaries |
| Drill-down | Not possible | Expand any summary to source messages or child summaries |
| Cross-session | No | All conversations in one DB, searchable |
| Data loss | Old summaries overwritten | Nothing lost — messages are immutable, summaries are superseded but kept |
