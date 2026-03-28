"""
run_baseline.py — Baseline inference script for the Customer Support OpenEnv.

Two modes:
  1. LLM Baseline  — Uses OpenAI API (requires OPENAI_API_KEY env var)
  2. Rule-Based    — Deterministic fallback, always reproducible

Usage:
  python baseline/run_baseline.py                        # LLM mode (needs OPENAI_API_KEY)
  python baseline/run_baseline.py --rule-based           # deterministic rule-based mode
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Any, Dict, List

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.environment import CustomerSupportEnv
from env.grader import grade_episode
from env.tasks import TASKS


# ---------------------------------------------------------------------------
# Rule-based baseline (no API key, always reproducible)
# ---------------------------------------------------------------------------

# Fixed deterministic responses per task per action
_RULE_BASED_POLICY: Dict[int, List[Dict[str, str]]] = {
    0: [
        {
            "action_type": "reply",
            "message": (
                "Thank you for reaching out! I sincerely apologize for the confusion. "
                "Your order is currently being processed and should arrive within "
                "3-5 business days. Please use your tracking number to monitor the "
                "delivery status. I'm happy to assist with anything else!"
            ),
        }
    ],
    1: [
        {
            "action_type": "refund",
            "message": (
                "I sincerely apologize for this inconvenience. I can see your order "
                "#ORD-7821 has not been delivered. I am immediately processing a full "
                "refund to your original payment method. You will receive a confirmation "
                "email within 24 hours and the refund should appear in 3–5 business days. "
                "Thank you for your patience and I'm sorry for the trouble."
            ),
        }
    ],
    2: [
        {
            "action_type": "refund",
            "message": (
                "I sincerely and deeply apologize for this completely unacceptable "
                "experience. A 3-week delay is inexcusable and I completely understand "
                "your frustration. I am immediately processing your full refund right now — "
                "you will receive a confirmation within minutes. I assure you this is "
                "our highest priority."
            ),
        },
        {
            "action_type": "escalate",
            "message": (
                "I am also immediately escalating your case to our senior supervisor "
                "who will contact you within the hour. You deserve better service "
                "and I sincerely apologize. Your concern is being treated with the "
                "utmost urgency."
            ),
        },
    ],
}


def run_rule_based_baseline() -> List[Dict[str, Any]]:
    """
    Run the deterministic rule-based baseline on all 3 tasks.
    Returns a list of grader results (score, turns_used, etc.)
    """
    env = CustomerSupportEnv()
    results = []

    for task in TASKS:
        task_id = task["id"]
        obs = env.reset(task_id=task_id)
        policy_steps = _RULE_BASED_POLICY.get(task_id, [])

        done = False
        for step_action in policy_steps:
            if done:
                break
            from env.models import Action
            action = Action.model_validate(step_action)
            result = env.step(action)
            done = result.done

        # Grade the episode
        env_state = env.state().model_dump()
        grade = grade_episode(task_id, env_state["conversation_history"], env_state)
        results.append(grade)

    return results


# ---------------------------------------------------------------------------
# LLM-powered baseline (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

def run_llm_baseline() -> List[Dict[str, Any]]:
    """
    Run an LLM-powered baseline agent using the OpenAI API.
    Reads OPENAI_API_KEY from environment variables.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Set it with: set OPENAI_API_KEY=sk-... (Windows) or export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    env = CustomerSupportEnv()
    results = []

    SYSTEM_PROMPT = """You are an expert customer support agent. 
You will receive a customer message and must respond with a JSON action.

Available actions:
- reply: Send a helpful message to the customer
- refund: Process a refund (also include a message explaining this)
- escalate: Escalate to a human supervisor (also include a message)
- ask_info: Request more information from the customer

ALWAYS respond with valid JSON in this exact format:
{
    "action_type": "<reply|refund|escalate|ask_info>",
    "message": "<your response to the customer>"
}

Be empathetic, professional, and concise. Choose the most appropriate action."""

    for task in TASKS:
        task_id = task["id"]
        print(f"\n--- Task {task_id} ({task['difficulty'].upper()}) ---")

        obs = env.reset(task_id=task_id)
        done = False
        turn = 0

        while not done and turn < task["max_turns"]:
            turn += 1

            # Build conversation context for LLM
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in obs.conversation_history:
                role = "user" if h["role"] == "customer" else "assistant"
                messages.append({"role": role, "content": h["content"]})

            messages.append({
                "role": "user",
                "content": (
                    f"Customer says: {obs.customer_message}\n"
                    f"Customer sentiment: {obs.sentiment}\n"
                    f"Issue type: {obs.issue_type}\n"
                    "What is your action? Respond with JSON only."
                ),
            })

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                action_json = json.loads(response.choices[0].message.content)
                action_type = action_json.get("action_type", "reply")
                message = action_json.get("message", "Thank you for contacting us.")
                print(f"  Turn {turn}: {action_type} — {message[:80]}...")

                from env.models import Action
                action = Action.model_validate({"action_type": action_type, "message": message})
                step_result = env.step(action)
                obs = step_result.observation
                done = step_result.done
                print(f"  Reward: {step_result.reward.value}")

            except Exception as e:
                print(f"  ERROR on turn {turn}: {e}")
                # Fallback: reply with a generic message
                from env.models import Action
                action = Action.model_validate({
                    "action_type": "reply",
                    "message": "I sincerely apologize for the inconvenience. Let me help you resolve this right away.",
                })
                step_result = env.step(action)
                obs = step_result.observation
                done = step_result.done

        # Grade episode
        env_state = env.state().model_dump()
        grade = grade_episode(task_id, env_state["conversation_history"], env_state)
        results.append(grade)
        print(f"  SCORE: {grade['score']}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict[str, Any]], baseline_type: str) -> None:
    print("\n" + "=" * 60)
    print(f"  BASELINE RESULTS — {baseline_type.upper()}")
    print("=" * 60)
    print(f"  {'Task':<6} {'Difficulty':<12} {'Score':<8} {'Turns'}")
    print(f"  {'-'*6} {'-'*12} {'-'*8} {'-'*6}")
    total = 0.0
    for r in results:
        print(f"  {r['task_id']:<6} {r['difficulty']:<12} {r['score']:<8.4f} {r['turns_used']}")
        total += r["score"]
    avg = total / len(results) if results else 0
    print(f"  {'-'*6} {'-'*12} {'-'*8}")
    print(f"  {'AVG':<6} {'':<12} {avg:<8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Support OpenEnv Baseline Agent")
    parser.add_argument(
        "--rule-based",
        action="store_true",
        help="Use the deterministic rule-based agent instead of the LLM agent.",
    )
    args = parser.parse_args()

    if args.rule_based:
        print("Running RULE-BASED baseline (deterministic, no API key needed)...")
        results = run_rule_based_baseline()
        print_results_table(results, "rule-based")
    else:
        print("Running LLM baseline with OpenAI (gpt-4o-mini)...")
        print("Set OPENAI_API_KEY environment variable if not already set.")
        results = run_llm_baseline()
        print_results_table(results, "llm (gpt-4o-mini)")

    print("\nBaseline complete. Results are reproducible.")
