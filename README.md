# Content Moderation RL Environment

A **Meta AI Hackathon OpenEnv benchmark** that models **social media moderation as a sequential reinforcement learning problem**.

Instead of classifying posts independently, the agent acts like a real trust-and-safety moderator: it reviews posts one by one, selects a moderation action, receives dense reward feedback, and must remain **accurate, fair, and behaviorally consistent across an entire session**.

This makes the benchmark significantly closer to **real platform safety workflows** than standard toxicity datasets.

---

##Why This Project Stands Out
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

## FastAPI + OpenEnv API
The environment is served through **FastAPI** and is fully **OpenEnv compatible**.

### Endpoints
- `GET /ping` → health + available tasks
- `POST /reset` → start new moderation episode
- `POST /step` → submit moderation action
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

## Why Judges Will Care
This project is not just a moderation demo.
It introduces a **benchmarking framework for evaluating sequential safety reasoning in LLM agents**.

It measures:
- moderation quality
- fairness under repeated exposure
- escalation calibration
- adversarial robustness
- behavioral consistency
- reward optimization under constraints

This framing makes it useful for:
- RL research
- LLM agent evaluation
- trust & safety simulation
- reward-model benchmarking
- human-AI moderation workflows

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
