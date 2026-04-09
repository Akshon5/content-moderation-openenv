---
title: Content Moderation OpenEnv
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ Content Moderation RL Environment


A **Meta AI Hackathon OpenEnv benchmark** that models **social media moderation as a sequential reinforcement learning problem**.

Instead of classifying posts independently, the agent acts like a real trust-and-safety moderator: it reviews posts one by one, selects a moderation action, receives dense reward feedback, and must remain **accurate, fair, and behaviorally consistent across an entire session**.

This makes the benchmark significantly closer to **real platform safety workflows** than standard toxicity datasets.

---

## Why This Project Stands Out
Most moderation benchmarks are static classification datasets.

This project evaluates whether an LLM or RL policy can perform:

- long-horizon moderation decisions
- uncertainty-aware escalation
- contextual fairness
- adversarial content handling
- policy consistency across repeated steps
- reward-driven decision optimization
- robust fallback behavior during API failures

The environment is intentionally designed to resemble **real-world trust & safety pipelines used by large social platforms**.

---

## Action Space
The agent selects **one of four discrete actions**:

- `approve` → safe content
- `warn` → borderline content, warn the author
- `escalate` → send to human moderator
- `remove` → clear violation

This mirrors realistic moderation workflows where the safest decision is not always removal.

---

## Observation Space
Each step includes:

- `post_id`
- `content`
- `author_history`
- `reported_count`
- `platform_context`
- `step_number`
- `total_steps`
- `current_score`
- `actions_taken`
- `recent_feedback`

The policy therefore reasons using both **local post content** and **session-level moderation history**.

---

## Dense Reward Design
The environment uses **step-wise dense reward shaping**.

### Positive rewards
- correct `approve` → +1.0
- correct `warn` → +0.8
- correct `escalate` → +0.7
- correct `remove` → +1.0

### Partial credit
Safer near-miss actions are still rewarded:
- escalate instead of remove
- warn instead of escalate
- remove instead of escalate

### Strong penalties
- harmful content approved → **-1.5**
- clean content removed → **-1.0**
- repeated same-action loops → penalty
- escalate-everything lazy policy → penalty
- overconfident wrong decisions → scaled penalty

This reward system encourages **safe, calibrated, human-like moderation behavior**.

---

## 3-Level Benchmark Curriculum
The benchmark is split into a realistic **difficulty ladder**.

### `task_easy`
Obvious moderation cases:
- safe benign posts
- direct harassment
- explicit spam
- dangerous misinformation
- clear hate speech
- explicit children-platform violations

**Goal:** establish baseline policy competence.

---

### `task_medium`
Context-sensitive reasoning:
- sarcasm
- subtle harassment
- political rhetoric
- suspicious but legitimate promotions
- repeat-offender fairness traps
- health anecdotes vs misinformation

**Goal:** evaluate nuanced moderation judgment.

---

### `task_hard`
Adversarial trust-and-safety benchmark:
- coded hate speech
- dog whistles
- satire vs genuine threats
- zero-width obfuscation
- brigading false reports
- severe zero-report violations
- self-harm coded phrasing
- political criticism vs harassment
- multi-policy conflicts

**Goal:** test robust reasoning beyond keyword matching.

---

## Baseline Results

The following deterministic baselines were obtained from local task-level sanity runs:

- `task_easy` and `task_medium`: **always approve**
- `task_hard`: **always escalate**

| Task        | Accuracy | Total Reward | Baseline Policy |
|-------------|----------|--------------|-----------------|
| task_easy   | 30%      | -8.76        | always approve  |
| task_medium | 40%      | -5.80        | always approve  |
| task_hard   | 20%      | -0.10        | always escalate |

These baselines intentionally use simple fixed-action policies to validate:
- reward shaping behavior
- curriculum difficulty progression
- anti-loop penalties
- cautious fallback handling

A stronger LLM policy is expected to significantly outperform these naive baselines.

## LLM Agent Results (Qwen/Qwen2.5-72B-Instruct)

Results from a full inference run using `inference.py`:

| Task        | Accuracy | Total Reward | Duration |
|-------------|----------|--------------|----------|
| task_easy   | 100%     | 11.95        | 30.87s   |
| task_medium | 50%      | 5.93         | 64.37s   |
| task_hard   | 80%      | 15.20        | 69.53s   |

Total wall time: **164.81s** (well within the 20-minute budget).

---

## Setup

```bash
pip install -r requirements.txt
```

---

## FastAPI + OpenEnv API
The environment is served through **FastAPI** and is fully **OpenEnv compatible**.

### Endpoints
- `GET /ping` → health + available tasks
- `POST /reset` → start new moderation episode
- `POST /step` → submit moderation action
- `GET /state` → full internal episode state
- `GET /summary` → episode accuracy and reward breakdown
- `GET /docs` → Swagger UI for testing

### Run locally
```bash
python app.py
```

Open:
```text
http://localhost:7860/docs
```

---

## LLM Inference Runner
`inference.py` runs an **OpenAI-compatible client** across all three tasks.

### Environment variables
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
```

### Run
```bash
python inference.py
```

### Features
- strict `[START] [STEP] [END]` OpenEnv logging
- retry on malformed JSON
- safe fallback escalation on API failures
- runtime budget awareness
- final `results.json` export
- per-task accuracy + reward reporting

---

## Docker Support
Build:
```bash
docker build -t moderation-env .
```

Run:
```bash
docker run -p 7860:7860 moderation-env
```

---

## Tech Stack
- Python 3.11
- FastAPI
- Pydantic
- OpenAI-compatible APIs
- Docker
- OpenEnv

---

## Submission Highlights
OpenEnv compliant  
3-tier curriculum benchmark  
Dense RL reward shaping  
FastAPI served  
Dockerized  
LLM inference runner  
Robust API-failure fallback  
Automatic results export  

This project reframes **content moderation as a realistic sequential decision benchmark**, making it highly relevant for next-generation LLM trust-and-safety systems.
