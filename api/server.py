"""
server.py — FastAPI server exposing all required OpenEnv endpoints.

Endpoints:
  GET  /            — health check
  POST /reset       — start new episode
  POST /step        — agent takes one action
  GET  /state       — current environment state
  GET  /tasks       — list all tasks + action schema
  GET  /grader      — grade the current (completed) episode
  GET  /baseline    — run baseline agent on all 3 tasks and return scores
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is on sys.path when running from api/ or root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.environment import CustomerSupportEnv
from env.grader import grade_episode
from env.models import Action
from env.tasks import list_tasks_summary

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Support OpenEnv",
    description=(
        "A real-world OpenEnv environment that simulates a customer support "
        "automation system. AI agents learn to resolve customer issues through "
        "the standard reset() / step() / state() interface."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (thread-safe for single-user HF Space)
_env = CustomerSupportEnv()
_last_task_id: Optional[int] = None

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = 0


class StepRequest(BaseModel):
    action_type: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def health_check() -> Dict[str, str]:
    """Returns 200 OK — confirms the service is running."""
    return {"status": "ok", "service": "customer-support-openenv", "version": "1.0.0"}


@app.post("/reset", summary="Start a new episode")
def reset(request: ResetRequest) -> Dict[str, Any]:
    """
    Initialise the environment for the given task_id (0=easy, 1=medium, 2=hard).
    Returns the initial Observation.
    """
    global _last_task_id
    try:
        obs = _env.reset(task_id=request.task_id)
        _last_task_id = request.task_id
        return {"observation": obs.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", summary="Agent takes one action")
def step(request: StepRequest) -> Dict[str, Any]:
    """
    Apply the agent's action. Returns observation, reward, done, info.
    Call /reset first to initialise an episode.
    """
    try:
        action = Action(
            action_type=request.action_type,  # type: ignore[arg-type]
            message=request.message,
        )
        result = _env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state", summary="Get current environment state")
def state() -> Dict[str, Any]:
    """Returns the full internal environment state."""
    try:
        return _env.state().model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/tasks", summary="List all tasks and action schema")
def tasks() -> Dict[str, Any]:
    """Returns all task definitions including action schema fields."""
    return {
        "tasks": list_tasks_summary(),
        "action_schema": {
            "action_type": {
                "type": "string",
                "enum": ["reply", "refund", "escalate", "ask_info"],
                "description": "The type of action the agent takes.",
            },
            "message": {
                "type": "string",
                "description": "The agent's response text.",
            },
        },
        "total_tasks": 3,
    }


@app.get("/grader", summary="Grade a completed episode")
def grader() -> Dict[str, Any]:
    """
    Returns the grader score for the most recent completed episode.
    Must call /reset and complete at least one /step first.
    """
    try:
        env_state = _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    state_dict = env_state.model_dump()
    task_id = state_dict["task_id"]
    history = state_dict["conversation_history"]

    result = grade_episode(task_id, history, state_dict)
    return {
        "grader_result": result,
        "score": result["score"],
        "episode_done": state_dict["done"],
    }


@app.get("/baseline", summary="Run baseline agent on all 3 tasks")
def baseline() -> Dict[str, Any]:
    """
    Runs the built-in rule-based baseline agent (no OpenAI key needed for this endpoint).
    Returns reproducible scores for all 3 tasks.

    For the OpenAI-powered baseline, run: python baseline/run_baseline.py
    """
    from baseline.run_baseline import run_rule_based_baseline  # lazy import

    try:
        scores = run_rule_based_baseline()
        return {
            "baseline_type": "rule_based",
            "note": "For LLM baseline use: python baseline/run_baseline.py with OPENAI_API_KEY set.",
            "scores": scores,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {e}")
