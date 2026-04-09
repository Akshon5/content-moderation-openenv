"""
inference.py — Content Moderation RL Agent
===========================================
Runs an LLM agent through all 3 moderation tasks using an OpenAI-compatible
API. Reads configuration from environment variables, logs every step, and
completes all tasks within the 20-minute budget.

Environment Variables:
    API_BASE_URL   — Base URL for OpenAI-compatible endpoint (required)
    MODEL_NAME     — Model identifier to use (required)
    OPENAI_API_KEY — API key (required)

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o"
    export OPENAI_API_KEY="sk-..."
    python inference.py

Log format (strict — matches OpenEnv spec):
    [START] task=<name> posts=<n> timestamp=<iso>
    [STEP]  task=<name> step=<n>/<total> post_id=<id> decision=<action>
    [END]   task=<name> accuracy=<f> total_reward=<f> duration=<s>s
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables — matches pre-submission checklist exactly
# Defaults set ONLY for API_BASE_URL and MODEL_NAME, NOT for HF_TOKEN
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME   = os.getenv("MODEL_NAME",   "<your-active-model>")
HF_TOKEN     = os.getenv("HF_TOKEN")          # No default — must be set at runtime
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional — only if using from_docker_image()

# ---------------------------------------------------------------------------
# Import environment + tasks
# ---------------------------------------------------------------------------

from environment import Action, ModerationAction, Observation

# Tasks are imported lazily so a missing task file causes a clear error
def _load_tasks():
    try:
        from tasks.task_easy   import make_env as make_easy
        from tasks.task_medium import make_env as make_medium
        from tasks.task_hard   import make_env as make_hard
        return [
            ("task_easy",   make_easy),
            ("task_medium", make_medium),
            ("task_hard",   make_hard),
        ]
    except ImportError as e:
        print(f"[ERROR] Could not import task files: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_config() -> Dict[str, str]:
    """Read and validate required environment variables."""
    cfg = {
        "api_base_url": API_BASE_URL.strip(),
        "model_name":   MODEL_NAME.strip(),
        "api_key":      (HF_TOKEN or "").strip(),
    }
    # Fail fast if any required value is missing or still a placeholder
    missing = [
        k for k, v in cfg.items()
        if not v or v.startswith("<your-")
    ]
    if missing:
        print(
            f"[ERROR] Missing required environment variables: "
            f"{[k.upper() for k in missing]}",
            file=sys.stderr,
        )
        sys.exit(1)
    return cfg


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert social media content moderator. Your job is to
review posts and choose the single best moderation action.

ACTIONS:
  approve  — post is safe, allow it through
  warn     — borderline content, flag the author with a warning
  escalate — ambiguous or complex, needs human reviewer
  remove   — clear policy violation, take down immediately

POLICY GUIDELINES:
  • Hate speech, harassment, explicit threats → remove
  • Misinformation that causes real-world harm → remove
  • Coded language, dog-whistles, adversarial phrasing → remove
  • Ambiguous satire, complex cultural context → escalate
  • Minor borderline content, repeat-offender borderline → warn
  • Clean content, even from repeat offenders → approve
  • High report counts alone do NOT justify removal — read the content

CONTEXT RULES:
  • children platform: stricter — adult-adjacent content → remove
  • news platform: political speech gets more latitude
  • adult platform: explicit content may be allowed
  • general platform: standard community guidelines apply

AUTHOR HISTORY:
  • clean          — no prior violations, benefit of the doubt
  • warned         — has received warnings, less benefit of the doubt
  • repeat_offender — multiple violations, but still judge THIS post fairly

Respond ONLY with valid JSON in this exact format (no markdown, no explanation):
{
  "decision": "<approve|warn|escalate|remove>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence max>"
}"""


def _build_user_prompt(obs: Observation) -> str:
    """Convert an Observation into a user message for the LLM."""
    actions_summary = ", ".join(
        f"{k}={v}" for k, v in obs.actions_taken.items() if v > 0
    ) or "none yet"

    feedback_line = (
        f"Last feedback: {obs.recent_feedback}\n" if obs.recent_feedback else ""
    )

    return f"""Review this post and decide the moderation action.

POST ID: {obs.post_id}
CONTENT: {obs.content}

CONTEXT:
  Author history  : {obs.author_history}
  Reports received: {obs.reported_count}
  Platform        : {obs.platform_context}

SESSION INFO:
  Step            : {obs.step_number + 1} / {obs.total_steps}
  Current score   : {obs.current_score:.2%}
  Actions so far  : {actions_summary}
{feedback_line}
Respond with JSON only."""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(
    client: OpenAI,
    model: str,
    obs: Observation,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Call the LLM and parse its JSON response.
    Retries on malformed JSON up to max_retries times.
    Returns a dict with keys: decision, confidence, reasoning.
    """
    user_prompt = _build_user_prompt(obs)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,        # Low temp for consistent decisions
                max_tokens=150,
                timeout=30,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if model wraps in ```json ... ```
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)

            # Validate decision field
            decision_str = parsed.get("decision", "").lower()
            if decision_str not in {a.value for a in ModerationAction}:
                raise ValueError(f"Invalid decision: {decision_str!r}")

            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return {
                "decision":   decision_str,
                "confidence": confidence,
                "reasoning":  str(parsed.get("reasoning", ""))[:200],
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1)
            continue

        except Exception as e:
            # Network / API error — wait longer before retry
            last_error = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            continue

    # All retries exhausted — fall back to escalate (safest default)
    print(
        f"    [WARN] LLM parse failed after {max_retries} attempts "
        f"({last_error}). Defaulting to escalate.",
        file=sys.stderr,
    )
    return {"decision": "escalate", "confidence": 0.1, "reasoning": "parse_error_fallback"}


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    make_env_fn,
    client: OpenAI,
    model: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run one full episode of a task. Logs [START], [STEP]*, [END].
    Returns summary dict.
    """
    env = make_env_fn(seed=seed)
    obs = env.reset()
    task_start = time.time()

    # ------------------------------------------------------------------
    # [START] log — must match: [START] task=<name> env=<benchmark> model=<model>
    # ------------------------------------------------------------------
    print(
        f"[START] task={task_name} env=content-moderation model={model}",
        flush=True,
    )

    step_results = []

    while True:
        if obs.post_id == "EPISODE_DONE":
            break

        step_num = obs.step_number + 1

        # LLM decision
        llm_out = _call_llm(client, model, obs)
        decision_enum = ModerationAction(llm_out["decision"])

        action = Action(
            post_id=obs.post_id,
            decision=decision_enum,
            confidence=llm_out["confidence"],
            reasoning=llm_out["reasoning"],
        )

        obs, reward, done, info = env.step(action)

        # ------------------------------------------------------------------
        # [STEP] log — must match:
        # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
        # ------------------------------------------------------------------
        error_val = info.get("error") or "null"
        done_val = str(done).lower()
        print(
            f"[STEP] step={step_num} action={llm_out['decision']} "
            f"reward={reward.total:.2f} done={done_val} error={error_val}",
            flush=True,
        )

        step_results.append({
            "step":           step_num,
            "post_id":        info["post_id"],
            "decision":       llm_out["decision"],
            "correct_action": info["correct_action"],
            "correct":        info.get("was_correct", False),
            "reward":         reward.total,
            "confidence":     llm_out["confidence"],
            "reasoning":      llm_out["reasoning"],
            "feedback":       reward.feedback,
        })

        if done:
            break

    summary = env.summary()
    duration = round(time.time() - task_start, 2)

    # ------------------------------------------------------------------
    # [END] log — must match:
    # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
    # ------------------------------------------------------------------
    score = summary["accuracy"]
    success = score > 0.0
    steps_taken = len(step_results)
    rewards_str = ",".join(f"{s['reward']:.2f}" for s in step_results)
    print(
        f"[END] success={str(success).lower()} steps={steps_taken} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
    print()

    return {
        "task_name":       task_name,
        "accuracy":        summary["accuracy"],
        "correct":         summary["correct_decisions"],
        "total_posts":     summary["total_posts"],
        "total_reward":    summary["total_reward"],
        "actions":         summary["actions_distribution"],
        "duration_sec":    duration,
        "steps":           step_results,
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main():
    global_start = time.time()

    # --- Config ---
    cfg    = _get_config()
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["api_base_url"])
    model  = cfg["model_name"]
    tasks  = _load_tasks()

    print("=" * 70)
    print(f"Content Moderation RL Agent")
    print(f"Model      : {model}")
    print(f"API Base   : {cfg['api_base_url']}")
    print(f"Tasks      : {[t[0] for t in tasks]}")
    print(f"Started at : {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    print()

    all_results = []
    BUDGET_SECONDS = 18 * 60  # 18-minute soft cap (20-min hard limit)

    for task_name, make_env_fn in tasks:
        elapsed = time.time() - global_start
        remaining = BUDGET_SECONDS - elapsed

        if remaining <= 0:
            print(
                f"[WARN] Budget exhausted before {task_name}. Skipping.",
                file=sys.stderr,
            )
            break

        result = run_task(
            task_name=task_name,
            make_env_fn=make_env_fn,
            client=client,
            model=model,
            seed=42,
        )
        all_results.append(result)

        elapsed = time.time() - global_start
        print(f"  ⏱  Elapsed: {elapsed:.1f}s / {BUDGET_SECONDS}s budget")
        print()

    # ------------------------------------------------------------------
    # Final summary across all tasks
    # ------------------------------------------------------------------
    total_elapsed = round(time.time() - global_start, 2)

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for r in all_results:
        print(
            f"  {r['task_name']:<15} "
            f"accuracy={r['accuracy']:.2%}  "
            f"reward={r['total_reward']:+.4f}  "
            f"time={r['duration_sec']}s"
        )

    if all_results:
        avg_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)
        avg_reward   = sum(r["total_reward"] for r in all_results) / len(all_results)
        print(f"\n  {'AVERAGE':<15} accuracy={avg_accuracy:.2%}  reward={avg_reward:+.4f}")

    print(f"\n  Total wall time: {total_elapsed}s")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Write results JSON for downstream analysis
    # ------------------------------------------------------------------
    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model":        model,
                "api_base_url": cfg["api_base_url"],
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                "total_elapsed_sec": total_elapsed,
                "tasks": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved → {output_path}")


if __name__ == "__main__":
    main()
