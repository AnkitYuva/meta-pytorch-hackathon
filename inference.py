"""
inference.py — Entry-point inference script for the Customer Support OpenEnv.

Output format (strictly per OpenEnv spec):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.models import Action
from env.environment import CustomerSupportEnv
from env.tasks import TASKS
from env.grader import grade_episode

# ---------------------------------------------------------------------------
# Mandatory Environment Variables (per spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK   = "customer_support_openenv"
TEMPERATURE = 0.2

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert customer support agent.
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

    Do not include any text outside the JSON object.
    Be empathetic, professional, and concise. Choose the most appropriate action.
    """
).strip()


# ---------------------------------------------------------------------------
# Structured logging — spec-compliant
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(client: OpenAI, obs) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for h in obs.history:
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
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
        )
        content = response.choices[0].message.content or "{}"
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception as exc:
        return {
            "action_type": "reply",
            "message": "I apologize for the inconvenience. Let me look into this for you right away.",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env    = CustomerSupportEnv()

    for task in TASKS:
        task_id   = task["id"]
        task_name = f"task_{task_id}"

        rewards: List[float] = []
        steps_taken = 0
        success     = False

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            obs  = env.reset(task_id=task_id)
            done = False

            for step in range(1, task["max_turns"] + 1):
                if done:
                    break

                action_dict = get_model_action(client, obs)
                error_msg   = action_dict.pop("error", None)

                action_type = action_dict.get("action_type", "reply")
                message     = action_dict.get("message", "I apologize for the inconvenience.")

                # Escape newlines so the formatted stdout line stays single-line
                safe_msg   = message[:40].replace("\n", " ").replace("\r", " ")
                action_str = f"{action_type}('{safe_msg}...')"

                action_obj = Action(action_type=action_type, message=message)

                try:
                    result     = env.step(action_obj)
                    reward_val = result.reward.value if result.reward else 0.0
                    done       = result.done
                    obs        = result.observation
                except Exception as e:
                    reward_val = 0.0
                    done       = True
                    error_msg  = str(e)

                rewards.append(reward_val)
                steps_taken = step

                log_step(
                    step=step,
                    action=action_str,
                    reward=reward_val,
                    done=done,
                    error=error_msg,
                )

                if done:
                    break

            # Grade episode
            try:
                env_state = env.state().model_dump()
                grade     = grade_episode(task_id, env_state["history"], env_state)
                score     = grade["score"]
                success   = score > 0.5
            except Exception:
                score   = sum(rewards) / len(rewards) if rewards else 0.0
                success = score > 0.5

        except Exception as e:
            print(f"[DEBUG] Episode error for task {task_id}: {e}", flush=True)

        finally:
            # [END] MUST always be emitted — even on exception
            log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
