#!/usr/bin/env python3
"""
Antigravity Interaction Chainer

Preserves conversation context across multiple Antigravity agent interactions.

Two strategies:
  1. previous_interaction_id  — link each new interaction to the prior one's ID
  2. full_history_input       — pass the entire conversation as a list of steps

Usage:
  python3 antigravity_chainer.py --strategy prev_id --task "Your task here"
  python3 antigravity_chainer.py --strategy full_history --task "Your task here"

The chainer maintains a conversation log in memory and can optionally
persist it to disk for cross-process continuity.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from google import genai

API_KEY = os.environ.get("GOOGLE_API_KEY", "AQ.Ab8RN6I-nXc9z7ca0P0AV3YJG2Od0A5FHlDgD8qJIg2iCNTOOQ")
AGENT = "antigravity-preview-05-2026"
ENVIRONMENT = "remote"

# Optional persistence path - alongside script in repo root
HISTORY_FILE = Path(__file__).parent / "conversation_history.json"


def load_history() -> list[dict]:
    """Load persisted conversation history from disk."""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_history(history: list[dict]) -> None:
    """Persist conversation history to disk."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def run_interaction_prev_id(
    client: genai.Client,
    input_text: str,
    previous_interaction_id: Optional[str],
    system_instruction: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Strategy 1: Use previous_interaction_id to chain interactions.

    Each call creates a fresh interaction but links it to the previous one,
    so the agent has full context from the prior conversation.
    """
    kwargs = dict(
        agent=AGENT,
        input=input_text,
        environment=ENVIRONMENT,
        store=True,
    )
    if previous_interaction_id:
        kwargs["previous_interaction_id"] = previous_interaction_id
    if system_instruction:
        kwargs["system_instruction"] = system_instruction

    interaction = client.interactions.create(**kwargs)
    print(f"[prev_id] Status: {interaction.status}")
    print(f"[prev_id] Interaction ID: {interaction.id}")
    print(f"[prev_id] Output:\n{interaction.output_text}")
    return interaction.output_text, interaction.id


def run_interaction_full_history(
    client: genai.Client,
    history: list[dict],
    current_input: str,
    system_instruction: Optional[str] = None,
) -> str:
    """
    Strategy 2: Pass the full conversation history as a list of steps.

    The input is a list of UserInputStepParam objects, each with type="user_input"
    and content containing the message. This sends the entire conversation
    in a single request.
    """
    steps = []
    for turn in history:
        steps.append({
            "type": "user_input",
            "content": [{"type": "text", "text": turn["content"]}],
        })
    # Add the current input as the final step
    steps.append({
        "type": "user_input",
        "content": [{"type": "text", "text": current_input}],
    })

    kwargs = dict(
        agent=AGENT,
        input=steps,
        environment=ENVIRONMENT,
        store=True,
    )
    if system_instruction:
        kwargs["system_instruction"] = system_instruction

    interaction = client.interactions.create(**kwargs)
    print(f"[full_history] Status: {interaction.status}")
    print(f"[full_history] Interaction ID: {interaction.id}")
    print(f"[full_history] Output:\n{interaction.output_text}")
    return interaction.output_text


def main():
    parser = argparse.ArgumentParser(description="Chain Antigravity agent interactions")
    parser.add_argument(
        "--strategy",
        choices=["prev_id", "full_history"],
        default="prev_id",
        help="Chaining strategy: prev_id (link via previous_interaction_id) or full_history (pass all turns as steps)",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="The task/input for this interaction",
    )
    parser.add_argument(
        "--system-instruction",
        default=None,
        help="Optional system instruction to prepend",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist conversation history to disk for cross-process continuity",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear persisted history before running",
    )
    args = parser.parse_args()

    if args.clear and HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
        print(f"Cleared history file: {HISTORY_FILE}")

    client = genai.Client(api_key=API_KEY)

    if args.strategy == "prev_id":
        # Load previous interaction ID from persisted history
        history = load_history()
        prev_id = None
        if history and "last_interaction_id" in history[-1]:
            prev_id = history[-1]["last_interaction_id"]
            print(f"Chaining from previous interaction: {prev_id}")

        output, interaction_id = run_interaction_prev_id(
            client,
            input_text=args.task,
            previous_interaction_id=prev_id,
            system_instruction=args.system_instruction,
        )

        if args.persist:
            history.append({
                "input": args.task,
                "output": output,
                "last_interaction_id": interaction_id,
            })
            save_history(history)
            print(f"History saved to {HISTORY_FILE}")

    elif args.strategy == "full_history":
        history = load_history()
        # Filter to just the conversation turns (not metadata)
        turns = [{"content": h["input"], "output": h.get("output", "")} for h in history if "input" in h]

        output = run_interaction_full_history(
            client,
            history=turns,
            current_input=args.task,
            system_instruction=args.system_instruction,
        )

        if args.persist:
            history.append({
                "input": args.task,
                "output": output,
            })
            save_history(history)
            print(f"History saved to {HISTORY_FILE}")


if __name__ == "__main__":
    main()
