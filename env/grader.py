"""
grader.py — Deterministic episode graders for the Customer Support OpenEnv.
Each grader returns a float in [0.0, 1.0] based on the full episode history.
"""

from __future__ import annotations
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _contains_any(text: str, keywords: List[str]) -> bool:
    """Case-insensitive keyword check."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _action_used(history: List[Dict[str, str]], action_type: str) -> bool:
    """Check if the agent ever used a specific action_type during the episode."""
    return any(
        turn.get("role") == "agent_action" and turn.get("action_type") == action_type
        for turn in history
    )


def _agent_messages(history: List[Dict[str, str]]) -> List[str]:
    """Extract all agent message strings from history."""
    return [
        turn.get("content", "")
        for turn in history
        if turn.get("role") in ("agent", "agent_action")
    ]


# ---------------------------------------------------------------------------
# Task 0 — Easy grader
# ---------------------------------------------------------------------------

def _grade_task_0(history: List[Dict[str, str]], final_state: Dict[str, Any]) -> float:
    """
    Task 0: Simple order status inquiry.
    Scoring:
      0.50 — agent used 'reply' action (not refund/escalate unnecessarily)
      0.30 — message contains order/status/tracking keywords
      0.20 — message contains polite/empathetic keywords
    """
    score = 0.0
    messages = _agent_messages(history)
    all_text = " ".join(messages)

    helpful_keywords = [
        "order", "track", "status", "ship", "deliver",
        "days", "update", "dispatch", "transit", "location",
    ]
    polite_keywords = [
        "sorry", "apologize", "understand", "happy to",
        "please", "thank", "help", "assist", "certainly",
    ]

    # +0.50 used 'reply' and did NOT use penalty actions for a simple FAQ
    if _action_used(history, "reply"):
        score += 0.50
        # Slight deduction if they escalated for a simple question
        if _action_used(history, "escalate") or _action_used(history, "refund"):
            score -= 0.15

    # +0.30 helpful keywords in response
    if _contains_any(all_text, helpful_keywords):
        score += 0.30

    # +0.20 polite tone
    if _contains_any(all_text, polite_keywords):
        score += 0.20

    return round(max(0.05, min(0.95, score)), 4)


# ---------------------------------------------------------------------------
# Task 1 — Medium grader
# ---------------------------------------------------------------------------

def _grade_task_1(history: List[Dict[str, str]], final_state: Dict[str, Any]) -> float:
    """
    Task 1: Missing package + refund request.
    Scoring:
      0.45 — used 'refund' OR 'ask_info' (appropriate action)
      0.30 — message is helpful (refund/investigation/verify keywords)
      0.25 — message is polite/empathetic
      -0.20 penalty — used 'escalate' without refunding (over-escalation)
    """
    score = 0.0
    messages = _agent_messages(history)
    all_text = " ".join(messages)

    helpful_keywords = [
        "refund", "process", "investigation", "verify", "confirm",
        "order", "money", "return", "reimburs", "received", "missing",
    ]
    polite_keywords = [
        "sorry", "apologize", "understand", "inconvenience",
        "please", "thank", "help", "frustrat", "concern",
    ]

    used_refund = _action_used(history, "refund")
    used_ask_info = _action_used(history, "ask_info")
    used_escalate = _action_used(history, "escalate")

    # +0.45 correct primary action
    if used_refund:
        score += 0.45
    elif used_ask_info:
        score += 0.35  # slightly less optimal but valid

    # +0.30 helpful message content
    if _contains_any(all_text, helpful_keywords):
        score += 0.30

    # +0.25 polite tone
    if _contains_any(all_text, polite_keywords):
        score += 0.25

    # -0.20 penalty: escalated without issuing refund (unnecessary escalation)
    if used_escalate and not used_refund:
        score -= 0.20

    return round(max(0.05, min(0.95, score)), 4)


# ---------------------------------------------------------------------------
# Task 2 — Hard grader
# ---------------------------------------------------------------------------

def _grade_task_2(history: List[Dict[str, str]], final_state: Dict[str, Any]) -> float:
    """
    Task 2: Angry customer — 3-week delay, demands refund + manager.
    Scoring:
      0.35 — used 'refund' action
      0.25 — used 'escalate' action (or mentioned escalation/manager)
      0.20 — empathy keywords in message
      0.20 — polite / professional tone
      -0.25 penalty — used 'ask_info' (tone-deaf for an angry customer)
    """
    score = 0.0
    messages = _agent_messages(history)
    all_text = " ".join(messages)

    empathy_keywords = [
        "sincerely sorry", "deeply apologize", "completely understand",
        "frustrat", "unacceptable", "right to be", "i hear you",
        "can imagine", "your concern", "serious", "immediate",
        "priority", "three weeks", "inexcusable",
    ]
    polite_keywords = [
        "apologize", "sorry", "understand", "assure",
        "please", "value", "important", "certainly", "immediately",
    ]
    escalation_keywords = [
        "manager", "supervisor", "escalat", "senior", "specialist",
        "team lead", "higher", "transfer",
    ]

    used_refund = _action_used(history, "refund")
    used_escalate = _action_used(history, "escalate")
    used_ask_info = _action_used(history, "ask_info")

    # +0.35 issued refund
    if used_refund:
        score += 0.35

    # +0.25 escalated OR mentioned escalation in message
    if used_escalate or _contains_any(all_text, escalation_keywords):
        score += 0.25

    # +0.20 empathy in message
    if _contains_any(all_text, empathy_keywords):
        score += 0.20

    # +0.20 professional/polite tone
    if _contains_any(all_text, polite_keywords):
        score += 0.20

    # -0.25 penalty: asking for info when customer is already very angry
    if used_ask_info:
        score -= 0.25

    return round(max(0.05, min(0.95, score)), 4)


# ---------------------------------------------------------------------------
# Public grading dispatch
# ---------------------------------------------------------------------------

_GRADERS = {
    0: _grade_task_0,
    1: _grade_task_1,
    2: _grade_task_2,
}


def grade_episode(
    task_id: int,
    history: List[Dict[str, str]],
    final_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Grade a completed episode for the given task.

    Returns:
        {
            "task_id": int,
            "difficulty": str,
            "score": float,          # strictly (0.0, 1.0) — never 0.0 or 1.0 exactly
            "turns_used": int,
        }
    """
    if task_id not in _GRADERS:
        raise ValueError(f"No grader registered for task_id={task_id}")

    from env.tasks import get_task  # local import to avoid circular deps

    task = get_task(task_id)
    raw_score = _GRADERS[task_id](history, final_state)

    # Validator requires score strictly in (0, 1) — i.e. not 0.0 and not 1.0
    score = round(max(0.05, min(0.95, raw_score)), 4)
    turns_used = final_state.get("turn", 0)

    return {
        "task_id": task_id,
        "difficulty": task["difficulty"],
        "score": score,
        "turns_used": turns_used,
    }
