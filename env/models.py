"""
models.py — Typed Pydantic models for the OpenEnv Customer Support Environment.
All API request/response types are defined here.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """The action an AI agent takes in a single turn."""

    action_type: Literal["reply", "refund", "escalate", "ask_info"] = Field(
        ...,
        description=(
            "Type of action the agent chooses:\n"
            "  reply    – Respond with a message to the customer\n"
            "  refund   – Issue a refund\n"
            "  escalate – Escalate to a human agent / supervisor\n"
            "  ask_info – Ask the customer for missing information"
        ),
    )
    message: str = Field(
        ...,
        min_length=1,
        description="The text content of the agent's response or action.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Everything the AI agent can observe at any given step."""

    customer_message: str = Field(
        ..., description="The latest message from the customer."
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full turn-by-turn conversation history.",
    )
    sentiment: Literal["positive", "neutral", "frustrated", "angry"] = Field(
        ..., description="Estimated sentiment of the customer."
    )
    issue_type: str = Field(
        ..., description="Category of the customer issue (e.g. 'missing_order')."
    )
    turn: int = Field(..., description="Current turn number (1-indexed).")
    max_turns: int = Field(..., description="Maximum allowed turns for this episode.")
    task_id: int = Field(..., description="ID of the active task (0, 1, or 2).")
    task_difficulty: str = Field(..., description="Difficulty label: easy / medium / hard.")


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Granular breakdown of how the reward was computed."""

    intent_score: float = Field(0.0, description="Reward for choosing the correct action type.")
    helpfulness_score: float = Field(0.0, description="Reward for response relevance/helpfulness.")
    tone_score: float = Field(0.0, description="Reward for polite, empathetic tone.")
    penalty: float = Field(0.0, description="Penalty for wrong action or unhelpful message (negative).")


class Reward(BaseModel):
    """Total reward for one agent step, with partial-credit breakdown."""

    value: float = Field(..., description="Net reward for this step (can be negative).")
    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)


# ---------------------------------------------------------------------------
# Step result returned by step()
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Full result returned by step()."""

    observation: Observation
    reward: Reward
    done: bool = Field(..., description="True if the episode has ended.")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra diagnostic information (e.g. resolution status, turn count).",
    )


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Full internal state of the environment (returned by state())."""

    task_id: int
    task_difficulty: str
    task_description: str
    issue_type: str
    current_customer_message: str
    sentiment: str
    history: List[Dict[str, Any]]
    turn: int
    max_turns: int
    resolved: bool
    resolution_action: Optional[str] = None
    done: bool
    cumulative_reward: float
