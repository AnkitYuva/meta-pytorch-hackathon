"""
inference.py — Entry-point inference script for the Customer Support OpenEnv.
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

# Mandatory Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "customer_support_openenv"
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

    Be empathetic, professional, and concise. Choose the most appropriate action.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as exc:
        return {"action_type": "reply", "message": "I apologize for the inconvenience.", "error": str(exc)}

def main() -> None:
    # Use HF_TOKEN if provided, otherwise fallback to standard OpenAI key for local testing
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY", "")
    
    # Initialize the OpenAI client as strictly required
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    env = CustomerSupportEnv()

    for task in TASKS:
        task_id = task["id"]
        task_name = f"task_{task_id}"
        
        history_actions: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        success = False
        score = 0.0
        
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        
        try:
            obs = env.reset(task_id=task_id)
            done = False
            
            for step in range(1, task["max_turns"] + 1):
                if done:
                    break
                
                action_dict = get_model_action(client, obs)
                error_msg = action_dict.pop("error", None)
                
                action_type = action_dict.get("action_type", "reply")
                message = action_dict.get("message", "I apologize.")
                
                # Escape newlines so our formatted stdout doesn't break
                safe_msg = message[:30].replace("\n", " ").replace("\r", " ")
                action_str = f"{action_type}(\'{safe_msg}...\')"
                
                action_obj = Action(action_type=action_type, message=message)
                
                try:
                    result = env.step(action_obj)
                    reward_val = result.reward.value if result.reward else 0.0
                    done = result.done
                    obs = result.observation
                except Exception as e:
                    reward_val = 0.0
                    done = True
                    error_msg = str(e)
                    
                rewards.append(reward_val)
                steps_taken = step
                
                log_step(step=step, action=action_str, reward=reward_val, done=done, error=error_msg)
                
                if done:
                    break
                    
            # Grade episode using the environment's official grader
            env_state = env.state().model_dump()
            grade = grade_episode(task_id, env_state["history"], env_state)
            score = grade["score"]
            success = score > 0.5  # Assume score > 0.5 is success
            
        except Exception as e:
            print(f"[DEBUG] Env error: {e}", flush=True)
            score = sum(rewards) / len(rewards) if rewards else 0.0
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
