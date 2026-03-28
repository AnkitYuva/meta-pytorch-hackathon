"""
environment.py — Core CustomerSupportEnv class implementing the OpenEnv spec.
Exposes reset(), step(), state() as required.
"""

from __future__ import annotations
import re
from typing import Any, Dict

from env.models import (
    Action,
    EnvironmentState,
    Observation,
    Reward,
    RewardBreakdown,
    StepResult,
)
from env.tasks import get_task


# ---------------------------------------------------------------------------
# Sentiment scoring for tone analysis
# ---------------------------------------------------------------------------

_POLITE_KEYWORDS = [
    "please", "thank you", "sorry", "apologize", "understand",
    "happy to", "certainly", "appreciate", "assist", "help",
    "value", "assure", "sincerely", "deeply", "kindly",
]
_HELPFUL_KEYWORDS = [
    "refund", "order", "track", "deliver", "process", "resolve",
    "investigate", "confirm", "account", "status", "return",
    "replacement", "update", "inform", "action", "immediately",
    "priority", "manager", "supervisor", "escalat",
]


def _score_tone(message: str) -> float:
    """Return 0.0–0.2 based on polite/empathetic keyword presence."""
    msg_lower = message.lower()
    hits = sum(1 for kw in _POLITE_KEYWORDS if kw in msg_lower)
    return round(min(0.2, hits * 0.05), 4)


def _score_helpfulness(message: str) -> float:
    """Return 0.0–0.3 based on helpful/relevant keyword presence."""
    msg_lower = message.lower()
    hits = sum(1 for kw in _HELPFUL_KEYWORDS if kw in msg_lower)
    return round(min(0.30, hits * 0.06), 4)


def _detect_sentiment(text: str) -> str:
    """Simple rule-based sentiment detector."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["unacceptable", "outrageous", "!!!", "now!", "demand", "worst"]):
        return "angry"
    if any(w in text_lower for w in ["frustrated", "upset", "annoyed", "disappointed", "terrible"]):
        return "frustrated"
    if any(w in text_lower for w in ["happy", "thank", "great", "wonderful", "appreciate"]):
        return "positive"
    return "neutral"


# ---------------------------------------------------------------------------
# Main Environment class
# ---------------------------------------------------------------------------

class CustomerSupportEnv:
    """
    OpenEnv-compliant Customer Support Automation environment.

    An AI agent interacts with a simulated customer support scenario.
    The agent must choose the correct action type and craft a helpful,
    empathetic response. Rewards are partial and dense (not sparse).
    """

    # Correct action → base intent reward mapping
    _INTENT_REWARD: Dict[str, float] = {
        "reply": 0.30,
        "refund": 0.40,
        "escalate": 0.35,
        "ask_info": 0.25,
    }

    # Penalty for using wrong action type for this task
    _WRONG_ACTION_PENALTY = -0.50

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}
        self._task: Dict[str, Any] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: int = 0) -> Observation:
        """
        Start a new episode for the given task.
        Returns the initial Observation shown to the agent.
        """
        task = get_task(task_id)
        scenario = task["scenario"]

        self._task = task
        self._state = {
            "task_id": task_id,
            "task_difficulty": task["difficulty"],
            "task_description": task["description"],
            "issue_type": scenario["issue_type"],
            "current_customer_message": scenario["customer_message"],
            "sentiment": scenario["sentiment"],
            "conversation_history": [
                {
                    "role": "customer",
                    "content": scenario["customer_message"],
                    "action_type": None,
                }
            ],
            "turn": 1,
            "max_turns": task["max_turns"],
            "resolved": False,
            "resolution_action": None,
            "done": False,
            "cumulative_reward": 0.0,
        }
        self._initialized = True

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Apply the agent's action to the environment.
        Returns StepResult(observation, reward, done, info).
        """
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")
        if self._state["done"]:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        s = self._state
        task = self._task
        expected_actions: list = task["expected_actions"]

        # ---- Record agent action in history ---------------------------
        s["conversation_history"].append({
            "role": "agent_action",
            "action_type": action.action_type,
            "content": action.message,
        })

        # ---- Compute reward -------------------------------------------
        breakdown = RewardBreakdown()

        # 1. Intent correctness
        if action.action_type in expected_actions:
            breakdown.intent_score = self._INTENT_REWARD.get(action.action_type, 0.30)
        else:
            # Wrong action for this task
            breakdown.penalty += self._WRONG_ACTION_PENALTY

        # 2. Helpfulness of message
        breakdown.helpfulness_score = _score_helpfulness(action.message)

        # 3. Tone / politeness
        breakdown.tone_score = _score_tone(action.message)

        # 4. Empty or trivially short message penalty
        if len(action.message.strip()) < 20:
            breakdown.penalty += -0.20

        net_reward = (
            breakdown.intent_score
            + breakdown.helpfulness_score
            + breakdown.tone_score
            + breakdown.penalty
        )
        net_reward = round(max(-1.0, min(1.0, net_reward)), 4)

        reward = Reward(value=net_reward, breakdown=breakdown)
        s["cumulative_reward"] = round(s["cumulative_reward"] + net_reward, 4)

        # ---- Update resolution state ----------------------------------
        if action.action_type in expected_actions and not s["resolved"]:
            s["resolved"] = True
            s["resolution_action"] = action.action_type

        # ---- Check done ----------------------------------------------
        s["turn"] += 1
        if s["resolved"] or s["turn"] > s["max_turns"]:
            s["done"] = True

        # ---- Next customer message (simple follow-up simulation) ------
        next_msg = self._generate_followup(action, s)
        if next_msg and not s["done"]:
            s["current_customer_message"] = next_msg
            s["conversation_history"].append(
                {"role": "customer", "content": next_msg, "action_type": None}
            )
            s["sentiment"] = _detect_sentiment(next_msg)

        observation = self._build_observation()

        info = {
            "turn": s["turn"] - 1,
            "max_turns": s["max_turns"],
            "resolved": s["resolved"],
            "resolution_action": s["resolution_action"],
            "cumulative_reward": s["cumulative_reward"],
        }

        return StepResult(
            observation=observation,
            reward=reward,
            done=s["done"],
            info=info,
        )

    def state(self) -> EnvironmentState:
        """Return the full internal state of the environment."""
        if not self._initialized:
            raise RuntimeError("Call reset() before state().")
        s = self._state
        return EnvironmentState(
            task_id=s["task_id"],
            task_difficulty=s["task_difficulty"],
            task_description=s["task_description"],
            issue_type=s["issue_type"],
            current_customer_message=s["current_customer_message"],
            sentiment=s["sentiment"],
            conversation_history=s["conversation_history"],
            turn=s["turn"],
            max_turns=s["max_turns"],
            resolved=s["resolved"],
            resolution_action=s["resolution_action"],
            done=s["done"],
            cumulative_reward=s["cumulative_reward"],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        s = self._state
        return Observation(
            customer_message=s["current_customer_message"],
            conversation_history=[
                {"role": t["role"], "content": t["content"]}
                for t in s["conversation_history"]
            ],
            sentiment=s["sentiment"],
            issue_type=s["issue_type"],
            turn=s["turn"],
            max_turns=s["max_turns"],
            task_id=s["task_id"],
            task_difficulty=s["task_difficulty"],
        )

    def _generate_followup(self, action: Action, s: Dict[str, Any]) -> str:
        """Generate a realistic follow-up customer message based on agent action."""
        if s["done"]:
            return ""

        action_type = action.action_type
        task_id = s["task_id"]

        # Task-specific follow-ups
        if task_id == 0:
            if action_type == "reply":
                return "Thank you for the update! How long will delivery take from now?"
            return "I'm still waiting for a proper answer about my order."

        if task_id == 1:
            if action_type == "refund":
                return "Thank you. When will the refund appear in my account?"
            if action_type == "ask_info":
                return "My order number is #ORD-7821, placed on March 15th."
            return "I still haven't heard anything about my refund. Please help."

        if task_id == 2:
            if action_type == "refund":
                return "Fine. I want to know when exactly the refund will process and I still want to speak to a manager."
            if action_type == "escalate":
                return "Good. When will the manager contact me? And what about my refund?"
            return "This is still not resolved! I need BOTH a refund AND to speak to a manager RIGHT NOW."

        return ""
