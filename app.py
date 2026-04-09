"""
app.py — Content Moderation RL Environment API
===============================================
FastAPI server exposing the environment over HTTP.
OpenEnv-compliant endpoints: GET /ping, POST /reset, POST /step

Usage:
    uvicorn app:app --host 0.0.0.0 --port 7860

Endpoints:
    GET  /ping          — Health check, returns {"status": "ok"}
    POST /reset         — Start a new episode, returns first Observation
    POST /step          — Submit a moderation action, returns step result
    GET  /state         — Full internal episode state (debug)
    GET  /summary       — Human-readable episode summary
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import (
    Action,
    ContentModerationEnv,
    ModerationAction,
    Observation,
    Reward,
)

# ---------------------------------------------------------------------------
# Task registry — load all available tasks
# ---------------------------------------------------------------------------

def _load_task_registry() -> Dict[str, Any]:
    registry = {}
    try:
        from tasks.task_easy import make_env as make_easy
        registry["task_easy"] = make_easy
    except ImportError:
        pass
    try:
        from tasks.task_medium import make_env as make_medium
        registry["task_medium"] = make_medium
    except ImportError:
        pass
    try:
        from tasks.task_hard import make_env as make_hard
        registry["task_hard"] = make_hard
    except ImportError:
        pass
    return registry


TASK_REGISTRY = _load_task_registry()
DEFAULT_TASK  = "task_hard" if "task_hard" in TASK_REGISTRY else (
    next(iter(TASK_REGISTRY)) if TASK_REGISTRY else None
)

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Content Moderation RL Environment",
    description=(
        "OpenEnv-compliant API where an LLM agent acts as a social media "
        "content moderator. Reviews posts and makes moderation decisions "
        "guided by dense reward signals."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global environment state
# Session is single-user (one active episode at a time).
# For multi-user scale, replace with a session-keyed dict.
# ---------------------------------------------------------------------------

_env: Optional[ContentModerationEnv] = None


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = DEFAULT_TASK or "task_hard"
    seed: int = 42


class StepRequest(BaseModel):
    post_id:    str
    decision:   str   # approve | warn | escalate | remove
    confidence: float = 0.5
    reasoning:  Optional[str] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward:      Dict[str, Any]
    done:        bool
    info:        Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/ping", tags=["Health"])
def ping():
    """Health check — must return 200 for HuggingFace Spaces."""
    return {
        "status":          "ok",
        "available_tasks": list(TASK_REGISTRY.keys()),
        "active_task":     _env.task_name if _env else None,
    }


@app.post("/reset", response_model=Dict[str, Any], tags=["Environment"])
def reset(req: ResetRequest):
    """
    Start a new episode.

    - Initializes the environment with the requested task and seed
    - Returns the first Observation the agent should act on

    Args:
        task: One of task_easy | task_medium | task_hard
        seed: Random seed for post shuffling (default 42)
    """
    global _env

    if req.task not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown task '{req.task}'. "
                f"Available: {list(TASK_REGISTRY.keys())}"
            ),
        )

    make_env_fn = TASK_REGISTRY[req.task]
    _env = make_env_fn(seed=req.seed)
    obs  = _env.reset()

    return {
        "observation": obs.model_dump(),
        "task":        req.task,
        "seed":        req.seed,
        "message":     f"Episode started. {obs.total_steps} posts to moderate.",
    }


@app.post("/step", response_model=StepResponse, tags=["Environment"])
def step(req: StepRequest):
    """
    Submit one moderation decision and receive the next observation + reward.

    Args:
        post_id:    Must match the post_id from the current observation
        decision:   approve | warn | escalate | remove
        confidence: Agent's confidence in its decision (0.0 - 1.0)
        reasoning:  Optional explanation string
    """
    global _env

    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )

    state = _env.state()
    if state.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is already done. Call POST /reset to start a new one.",
        )

    # Validate decision
    try:
        decision_enum = ModerationAction(req.decision.lower())
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid decision '{req.decision}'. "
                f"Must be one of: approve, warn, escalate, remove"
            ),
        )

    # Validate confidence range
    confidence = max(0.0, min(1.0, req.confidence))

    # Build action
    action = Action(
        post_id=req.post_id,
        decision=decision_enum,
        confidence=confidence,
        reasoning=req.reasoning,
    )

    # Step the environment
    try:
        obs, reward, done, info = _env.step(action)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state", tags=["Debug"])
def get_state():
    """Returns full internal episode state. Useful for debugging."""
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    return _env.state().model_dump()


@app.get("/summary", tags=["Debug"])
def get_summary():
    """Returns human-readable episode summary with accuracy and reward breakdown."""
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    return _env.summary()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
