"""
tasks.py — Task definitions for the Customer Support OpenEnv environment.
Each task is a concrete scenario an AI agent must resolve.
Difficulty: easy → medium → hard.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------
    # TASK 0 — EASY
    # Simple FAQ: customer wants to know where their order is.
    # Expected: agent replies with order status info.
    # ------------------------------------------------------------------
    {
        "id": 0,
        "difficulty": "easy",
        "description": (
            "A customer is asking a simple FAQ question about their order status. "
            "The agent should reply with a helpful, polite message explaining "
            "the tracking situation."
        ),
        "scenario": {
            "customer_message": (
                "Hi, I placed an order 3 days ago and I have no idea where it is. "
                "Can you tell me where my order is?"
            ),
            "sentiment": "neutral",
            "issue_type": "order_status_inquiry",
        },
        "expected_actions": ["reply"],
        "grading_criteria": {
            "correct_action_types": ["reply"],
            "helpful_keywords": [
                "order", "track", "status", "ship", "deliver",
                "days", "update", "dispatch", "transit",
            ],
            "polite_keywords": [
                "sorry", "apologize", "understand", "happy to",
                "please", "thank", "help", "assist",
            ],
        },
        "max_turns": 3,
        "action_schema": {
            "action_type": "Literal['reply', 'refund', 'escalate', 'ask_info']",
            "message": "str",
        },
    },

    # ------------------------------------------------------------------
    # TASK 1 — MEDIUM
    # Missing package + refund request. Agent needs more context OR
    # proactively issue refund / ask for order details.
    # ------------------------------------------------------------------
    {
        "id": 1,
        "difficulty": "medium",
        "description": (
            "A customer reports they did not receive their package and wants a refund. "
            "The agent must either request the order ID / more details, or proceed "
            "to issue the refund if policy allows. Escalation is suboptimal unless "
            "genuinely necessary."
        ),
        "scenario": {
            "customer_message": (
                "Hello, I haven't received my package and it has been over a week. "
                "I want my money back. My order number is #ORD-7821."
            ),
            "sentiment": "frustrated",
            "issue_type": "missing_package_refund",
        },
        "expected_actions": ["refund", "ask_info"],
        "grading_criteria": {
            "correct_action_types": ["refund", "ask_info"],
            "helpful_keywords": [
                "refund", "process", "investigation", "verify", "confirm",
                "order", "#ORD", "money", "return", "reimburs",
            ],
            "polite_keywords": [
                "sorry", "apologize", "understand", "inconvenience",
                "please", "thank", "help", "frustrat",
            ],
            "penalty_actions": ["escalate"],
        },
        "max_turns": 4,
        "action_schema": {
            "action_type": "Literal['reply', 'refund', 'escalate', 'ask_info']",
            "message": "str",
        },
    },

    # ------------------------------------------------------------------
    # TASK 2 — HARD
    # Extremely angry customer: wants refund AND escalation, uses strong
    # language. Agent must handle emotional tone, issue refund, AND
    # escalate. Multi-condition grading.
    # ------------------------------------------------------------------
    {
        "id": 2,
        "difficulty": "hard",
        "description": (
            "An angry customer has been waiting 3 weeks for their order, demands "
            "an immediate refund AND escalation to a manager. They are using "
            "strong emotional language. The agent must de-escalate, issue the "
            "refund, and acknowledge the escalation — all while maintaining "
            "empathy and professionalism."
        ),
        "scenario": {
            "customer_message": (
                "This is absolutely UNACCEPTABLE! I have been waiting THREE WEEKS "
                "for my order and no one has helped me. I demand a full refund "
                "immediately and I want to speak to your manager NOW. "
                "This is the worst service I have ever experienced in my life!"
            ),
            "sentiment": "angry",
            "issue_type": "long_delay_refund_escalation",
        },
        "expected_actions": ["refund", "escalate"],
        "grading_criteria": {
            "correct_action_types": ["refund", "escalate"],
            "helpful_keywords": [
                "refund", "manager", "supervisor", "escalat",
                "three weeks", "immediately", "process", "priority",
            ],
            "empathy_keywords": [
                "sincerely sorry", "deeply apologize", "completely understand",
                "frustrat", "unacceptable", "right to be", "I hear you",
                "can imagine", "your concern",
            ],
            "polite_keywords": [
                "apologize", "sorry", "understand", "assure",
                "please", "value", "important",
            ],
            "penalty_actions": ["ask_info"],
        },
        "max_turns": 5,
        "action_schema": {
            "action_type": "Literal['reply', 'refund', 'escalate', 'ask_info']",
            "message": "str",
        },
    },
]

# ---------------------------------------------------------------------------
# Helper lookups
# ---------------------------------------------------------------------------

TASK_BY_ID: Dict[int, Dict[str, Any]] = {t["id"]: t for t in TASKS}


def get_task(task_id: int) -> Dict[str, Any]:
    """Return task dict by ID, raising ValueError if not found."""
    if task_id not in TASK_BY_ID:
        raise ValueError(
            f"task_id {task_id} not found. Valid IDs: {list(TASK_BY_ID.keys())}"
        )
    return TASK_BY_ID[task_id]


def list_tasks_summary() -> List[Dict[str, Any]]:
    """Return a lightweight summary of all tasks (no internal grading details)."""
    return [
        {
            "id": t["id"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "expected_actions": t["expected_actions"],
            "max_turns": t["max_turns"],
            "action_schema": t["action_schema"],
        }
        for t in TASKS
    ]
