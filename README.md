---
title: Customer Support OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# 🤖 Customer Support OpenEnv

> **Meta Hackathon 2025 — Round 1 Submission**  
> A real-world OpenEnv environment for training and evaluating AI agents on customer support tasks.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.ai)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📖 Environment Description

Customer support is one of the highest-value, highest-volume tasks in modern business — yet most agents fail when faced with real emotional complexity, edge cases, or multi-step resolutions.

This environment simulates a **customer support inbox** where an AI agent must:
- Understand the nature of a customer complaint
- Choose the correct action type (`reply`, `refund`, `escalate`, `ask_info`)
- Write a helpful, empathetic, professional response
- Handle multi-turn conversations and escalating emotional states

The environment provides **dense partial-credit rewards** rather than sparse end-of-episode signals, making it suitable for RL, behavioural cloning, and LLM evaluation.

---

## 🎬 Action Space

| Action | Description |
|--------|-------------|
| `reply` | Send a helpful response to the customer |
| `refund` | Process a full refund for the customer |
| `escalate` | Escalate the case to a human supervisor |
| `ask_info` | Request more information from the customer |

**Action Schema (JSON):**
```json
{
  "action_type": "reply | refund | escalate | ask_info",
  "message": "Your response text here"
}
```

---

## 👁️ Observation Space

Each step the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `customer_message` | `string` | Latest customer message |
| `conversation_history` | `list[dict]` | Full turn history (role + content) |
| `sentiment` | `string` | `positive / neutral / frustrated / angry` |
| `issue_type` | `string` | Category of issue (e.g. `missing_package_refund`) |
| `turn` | `int` | Current turn number |
| `max_turns` | `int` | Maximum turns for this episode |
| `task_id` | `int` | Active task ID |
| `task_difficulty` | `string` | `easy / medium / hard` |

---

## 🎯 Tasks

### Task 0 — Easy: Order Status Inquiry
**Customer message:** *"Hi, I placed an order 3 days ago and I have no idea where it is. Can you tell me where my order is?"*

- **Expected action:** `reply`
- **Grading:** Checks for tracking/status keywords + polite tone
- **Max turns:** 3
- **Typical score (rule-based baseline):** `0.90`

---

### Task 1 — Medium: Missing Package Refund
**Customer message:** *"Hello, I haven't received my package and it has been over a week. I want my money back. My order number is #ORD-7821."*

- **Expected action:** `refund` or `ask_info`
- **Grading:** Correct action type + helpful resolution message + empathy
- **Max turns:** 4
- **Typical score (rule-based baseline):** `0.74`

---

### Task 2 — Hard: Angry Customer — Refund + Escalation
**Customer message:** *"This is absolutely UNACCEPTABLE! I have been waiting THREE WEEKS for my order... I demand a full refund immediately and I want to speak to your manager NOW."*

- **Expected actions:** `refund` AND `escalate`
- **Grading:** Multi-condition — refund (0.35) + escalation (0.25) + empathy (0.20) + professionalism (0.20), penalty for `ask_info` (−0.25)
- **Max turns:** 5
- **Typical score (rule-based baseline):** `0.80`

---

## 🏆 Reward Function

Rewards are **dense and partial** — signal is provided at every step, not just on termination.

| Component | Weight | Condition |
|-----------|--------|-----------|
| Intent correctness | +0.25 to +0.40 | Chose correct action type for this task |
| Helpfulness | +0.00 to +0.30 | Message contains relevant keywords |
| Polite tone | +0.00 to +0.20 | Message contains empathetic language |
| Wrong action | −0.50 | Used incorrect action type |
| Empty message | −0.20 | Message shorter than 20 characters |

**Range:** `[−1.0, 1.0]` per step

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/reset` | Start a new episode `{"task_id": 0}` |
| `POST` | `/step` | Agent takes action `{"action_type": "reply", "message": "..."}` |
| `GET` | `/state` | Full internal environment state |
| `GET` | `/tasks` | All task definitions + action schema |
| `GET` | `/grader` | Grade the current episode (0.0–1.0) |
| `GET` | `/baseline` | Run rule-based baseline on all 3 tasks |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 📊 Baseline Scores

Tested with the built-in **rule-based deterministic agent** (no API key needed):

| Task | Difficulty | Score |
|------|-----------|-------|
| 0 | Easy | 0.9000 |
| 1 | Medium | 0.7400 |
| 2 | Hard | 0.8000 |
| **AVG** | | **0.8133** |

*LLM baseline (gpt-4o-mini) typically achieves 0.85–0.95 average.*

---

## 🛠️ Setup & Usage

### Local Development

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-support-openenv
cd customer-support-openenv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI key (optional — only needed for LLM baseline)
set OPENAI_API_KEY=sk-...        # Windows
export OPENAI_API_KEY=sk-...     # Linux/Mac

# 4. Start the server
uvicorn api.server:app --reload --port 7860

# 5. Open docs
# http://localhost:7860/docs
```

### Quick API Test
```bash
# Reset (start Task 0)
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{\"task_id\": 0}"

# Step (agent acts)
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
  -d "{\"action_type\": \"reply\", \"message\": \"Your order is on its way and will arrive in 2-3 days!\"}"

# Grade
curl http://localhost:7860/grader
```

### Run Baseline
```bash
# Deterministic (no API key)
python baseline/run_baseline.py --rule-based

# LLM (requires OPENAI_API_KEY)
python baseline/run_baseline.py
```

---

## 🐳 Docker

```bash
# Build
docker build -t customer-support-openenv .

# Run
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... customer-support-openenv

# Test
curl http://localhost:7860/
```

---

## 📁 Project Structure

```
MetaHackathon/
├── env/
│   ├── models.py        # Pydantic typed models (Action, Observation, Reward)
│   ├── environment.py   # Core env (reset/step/state + reward logic)
│   ├── tasks.py         # Task definitions (easy/medium/hard)
│   └── grader.py        # Deterministic episode graders
├── api/
│   └── server.py        # FastAPI app with all endpoints
├── baseline/
│   └── run_baseline.py  # Rule-based + LLM baseline agents
├── openenv.yaml         # OpenEnv metadata
├── Dockerfile           # Container definition
├── requirements.txt
└── README.md
```

---

## 🚀 Deploying to Hugging Face Spaces

1. Create a new Space → select **Docker** as SDK
2. Push this repo to the Space's Git remote
3. Set the `OPENAI_API_KEY` secret in Space Settings → Variables
4. Space auto-builds and serves on `https://huggingface.co/spaces/YOUR_USERNAME/customer-support-openenv`

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for the Meta × Hugging Face OpenEnv Hackathon 2025.*
