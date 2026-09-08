# Antigravity Chainer

Chain Antigravity agent interactions (`antigravity-preview-05-2026`) so context persists across separate API calls.

## Problem

The Antigravity agent is **stateless across separate interactions** — each `client.interactions.create()` call spawns a fresh sandbox container. There is no server-side session memory between calls.

## Solution

Two strategies for context chaining:

### 1. `prev_id` — Link via `previous_interaction_id` (recommended)

Pass the `id` from the previous interaction as `previous_interaction_id` in the next call. The agent receives the full prior conversation context server-side.

```python
# Turn 1
resp1 = client.interactions.create(agent="antigravity-preview-05-2026", input="What is 2+2?")
prev_id = resp1.id

# Turn 2 — context preserved
resp2 = client.interactions.create(
    agent="antigravity-preview-05-2026",
    input="Now multiply that by 3",
    previous_interaction_id=prev_id,
)
# resp2.output_text -> "4 × 3 = 12"
```

### 2. `full_history` — Embed conversation in input

Pass the entire conversation history as the `input` string on every call. No linking needed, but input grows with each turn.

```python
history = "User: What is 2+2?\nAgent: 2 + 2 = 4.\nUser: Now multiply that by 3"
resp = client.interactions.create(
    agent="antigravity-preview-05-2026",
    input=history,
)
```

## Usage

```bash
# First interaction (prev_id strategy)
python3 antigravity_chainer.py --strategy prev_id --task "What is 2+2?" --persist --clear

# Chained interaction (context preserved)
python3 antigravity_chainer.py --strategy prev_id --task "Now multiply that by 3" --persist

# Full history strategy
python3 antigravity_chainer.py --strategy full_history --task "What was the original question?" --persist
```

### Arguments

| Flag | Description |
|---|---|
| `--strategy` | `prev_id` (link via interaction ID) or `full_history` (embed history in input) |
| `--task` | The prompt to send to the agent |
| `--persist` | Save conversation history to `conversation_history.json` |
| `--clear` | Clear history file before running |

## Key SDK Details

- **SDK**: `google-genai` v2.14.0+ (must be >= 2.0.0 for Interactions API)
- **API key**: Uses `GOOGLE_API_KEY` env var (or `GEMINI_API_KEY`)
- **Agent**: `antigravity-preview-05-2026` — a managed agent, not a model
- **Environment**: `remote` — runs in a Google-hosted sandbox
- **Response**: `interaction.output_text` contains the concatenated text output
- **Interaction ID**: `interaction.id` — use this for `previous_interaction_id`

## Files

- `antigravity_chainer.py` — CLI tool with both strategies
- `conversation_history.json` — persisted interaction chain (created on `--persist`)
