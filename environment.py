"""
Content Moderation RL Environment
==================================
OpenEnv-compliant simulation where an agent acts as a social media content moderator.
The agent reviews posts and makes moderation decisions to maintain platform safety
while minimizing over-moderation.
"""

import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModerationAction(str, Enum):
    APPROVE   = "approve"    # Post is safe, allow it
    WARN      = "warn"       # Post is borderline, flag with a warning
    ESCALATE  = "escalate"   # Post needs human review
    REMOVE    = "remove"     # Post clearly violates policy, remove it


class ViolationType(str, Enum):
    NONE        = "none"
    SPAM        = "spam"
    HATE_SPEECH = "hate_speech"
    MISINFORMATION = "misinformation"
    HARASSMENT  = "harassment"
    EXPLICIT    = "explicit"
    BORDERLINE  = "borderline"   # Ambiguous — could go either way


class Severity(str, Enum):
    NONE     = "none"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Post(BaseModel):
    """A social media post to be moderated."""
    post_id: str
    content: str
    author_history: str = Field(
        description="Author's violation history: clean / warned / repeat_offender"
    )
    reported_count: int = Field(ge=0, description="Number of user reports on this post")
    platform_context: str = Field(
        description="Where the post appears: general / children / news / adult"
    )
    # Ground truth — hidden from agent observation, used only for grading
    true_violation: ViolationType
    true_severity: Severity
    correct_action: ModerationAction


class Observation(BaseModel):
    """What the agent sees at each step."""
    post_id: str
    content: str
    author_history: str
    reported_count: int
    platform_context: str
    step_number: int
    total_steps: int
    current_score: float = Field(ge=0.0, le=1.0)
    actions_taken: Dict[str, int] = Field(
        description="Count of each action taken so far in this episode"
    )
    recent_feedback: Optional[str] = Field(
        default=None,
        description="Feedback signal from the last decision (dense reward hint)"
    )


class Action(BaseModel):
    """The moderation decision made by the agent."""
    post_id: str
    decision: ModerationAction
    confidence: float = Field(ge=0.0, le=1.0, description="Agent's confidence in decision")
    reasoning: Optional[str] = Field(default=None, description="Optional explanation")


class Reward(BaseModel):
    """Step-wise reward breakdown."""
    total: float = Field(description="Total reward for this step")
    accuracy_reward: float = Field(description="Reward for correct moderation decision")
    confidence_bonus: float = Field(description="Bonus for high confidence on correct decisions")
    consistency_penalty: float = Field(description="Penalty for inconsistent moderation patterns")
    over_moderation_penalty: float = Field(description="Penalty for removing/escalating clean content")
    under_moderation_penalty: float = Field(description="Penalty for approving harmful content")
    feedback: str = Field(description="Human-readable explanation of reward breakdown")


class EpisodeState(BaseModel):
    """Full internal state of the environment."""
    task_name: str
    posts: List[Post]
    current_index: int
    done: bool
    total_reward: float
    actions_taken: Dict[str, int]
    step_rewards: List[float]
    decisions: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Reward Configuration
# ---------------------------------------------------------------------------

REWARD_CONFIG = {
    # Correct decision rewards
    "correct_remove":   +1.0,
    "correct_warn":     +0.8,
    "correct_escalate": +0.7,
    "correct_approve":  +1.0,

    # Partial credit (close but not exact)
    "escalate_when_remove": +0.4,   # Escalated instead of removing — cautious but safe
    "warn_when_escalate":   +0.3,   # Warned instead of escalating — underreacted slightly
    "remove_when_escalate": +0.5,   # Removed instead of escalating — too aggressive but safe

    # Wrong direction penalties
    "approve_harmful":  -1.5,       # Letting harmful content through — worst mistake
    "remove_clean":     -1.0,       # Removing clean content — over-moderation
    "warn_clean":       -0.3,       # Warning clean content — minor over-moderation

    # Confidence bonus/penalty
    "confidence_bonus_scale":  +0.2,  # Max bonus for high confidence + correct
    "confidence_penalty_scale": -0.1, # Penalty for high confidence + wrong

    # Behavioral penalties
    "loop_penalty":      -0.2,       # Penalty for always choosing same action
    "escalate_all_penalty": -0.1,    # Penalty per step if escalating everything
}


# ---------------------------------------------------------------------------
# Core Environment
# ---------------------------------------------------------------------------

class ContentModerationEnv:
    """
    OpenEnv-compliant Content Moderation Environment.

    The agent acts as a platform moderator reviewing a queue of posts
    and deciding: approve / warn / escalate / remove.

    Dense rewards guide the agent toward accurate, consistent moderation
    that neither over- nor under-moderates.
    """

    def __init__(self, task_name: str, posts: List[Post], seed: int = 42):
        self.task_name = task_name
        self.posts = posts
        self.seed = seed
        self._rng = random.Random(seed)
        self._state: Optional[EpisodeState] = None
        self.reset()

    # ------------------------------------------------------------------
    # OpenEnv Required Methods
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment to initial state. Returns first observation."""
        shuffled_posts = self.posts.copy()
        self._rng.shuffle(shuffled_posts)

        self._state = EpisodeState(
            task_name=self.task_name,
            posts=shuffled_posts,
            current_index=0,
            done=False,
            total_reward=0.0,
            actions_taken={action.value: 0 for action in ModerationAction},
            step_rewards=[],
            decisions=[],
        )
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one moderation decision.

        Args:
            action: The moderation Action taken by the agent

        Returns:
            (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_post = self._state.posts[self._state.current_index]

        # Validate post_id matches
        if action.post_id != current_post.post_id:
            raise ValueError(
                f"Action post_id '{action.post_id}' does not match "
                f"current post '{current_post.post_id}'"
            )

        # Calculate reward
        reward = self._calculate_reward(action, current_post)

        # Update state
        self._state.actions_taken[action.decision.value] += 1
        self._state.total_reward += reward.total
        self._state.step_rewards.append(reward.total)
        self._state.decisions.append({
            "post_id": current_post.post_id,
            "decision": action.decision.value,
            "correct_action": current_post.correct_action.value,
            "reward": reward.total,
            "correct": action.decision == current_post.correct_action,
        })
        self._state.current_index += 1

        # Check if episode is done
        done = self._state.current_index >= len(self._state.posts)
        self._state.done = done

        # Build info dict
        info = {
            "post_id": current_post.post_id,
            "correct_action": current_post.correct_action.value,
            "true_violation": current_post.true_violation.value,
            "true_severity": current_post.true_severity.value,
            "was_correct": action.decision == current_post.correct_action,
            "episode_score": self._calculate_episode_score() if done else None,
            "total_reward": self._state.total_reward,
            "step": self._state.current_index,
        }

        obs = self._get_observation(recent_feedback=reward.feedback) if not done else self._get_observation()
        return obs, reward, done, info

    def state(self) -> EpisodeState:
        """Returns the full internal state of the environment."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    # ------------------------------------------------------------------
    # Observation Builder
    # ------------------------------------------------------------------

    def _get_observation(self, recent_feedback: Optional[str] = None) -> Observation:
        s = self._state
        if s.current_index >= len(s.posts):
            # Episode over — return final dummy observation
            return Observation(
                post_id="EPISODE_DONE",
                content="",
                author_history="",
                reported_count=0,
                platform_context="",
                step_number=s.current_index,
                total_steps=len(s.posts),
                current_score=self._calculate_episode_score(),
                actions_taken=s.actions_taken.copy(),
                recent_feedback="Episode complete.",
            )

        current_post = s.posts[s.current_index]
        return Observation(
            post_id=current_post.post_id,
            content=current_post.content,
            author_history=current_post.author_history,
            reported_count=current_post.reported_count,
            platform_context=current_post.platform_context,
            step_number=s.current_index,
            total_steps=len(s.posts),
            current_score=self._calculate_episode_score(),
            actions_taken=s.actions_taken.copy(),
            recent_feedback=recent_feedback,
        )

    # ------------------------------------------------------------------
    # Reward Calculator
    # ------------------------------------------------------------------

    def _calculate_reward(self, action: Action, post: Post) -> Reward:
        cfg = REWARD_CONFIG
        decision = action.decision
        correct = post.correct_action
        is_correct = decision == correct

        accuracy_reward = 0.0
        over_mod_penalty = 0.0
        under_mod_penalty = 0.0
        feedback_parts = []

        # --- Accuracy reward ---
        if is_correct:
            accuracy_reward = cfg[f"correct_{decision.value}"]
            feedback_parts.append(f"✓ Correct decision: {decision.value}")
        else:
            # Partial credit and directional penalties
            if correct == ModerationAction.REMOVE:
                if decision == ModerationAction.ESCALATE:
                    accuracy_reward = cfg["escalate_when_remove"]
                    feedback_parts.append("~ Escalated instead of removing — cautious but acceptable")
                elif decision == ModerationAction.WARN:
                    under_mod_penalty = cfg["approve_harmful"] * 0.5
                    feedback_parts.append("✗ Warned instead of removing — significant under-moderation")
                elif decision == ModerationAction.APPROVE:
                    under_mod_penalty = cfg["approve_harmful"]
                    feedback_parts.append("✗ Approved harmful content — worst outcome")

            elif correct == ModerationAction.ESCALATE:
                if decision == ModerationAction.REMOVE:
                    accuracy_reward = cfg["remove_when_escalate"]
                    feedback_parts.append("~ Removed instead of escalating — over-aggressive but safe")
                elif decision == ModerationAction.WARN:
                    accuracy_reward = cfg["warn_when_escalate"]
                    feedback_parts.append("~ Warned instead of escalating — slightly under-reacted")
                elif decision == ModerationAction.APPROVE:
                    under_mod_penalty = cfg["approve_harmful"] * 0.7
                    feedback_parts.append("✗ Approved content that needed escalation")

            elif correct == ModerationAction.WARN:
                if decision == ModerationAction.REMOVE:
                    over_mod_penalty = cfg["remove_clean"]
                    feedback_parts.append("✗ Removed content that only needed a warning — over-moderation")
                elif decision == ModerationAction.ESCALATE:
                    accuracy_reward = cfg["warn_when_escalate"]
                    feedback_parts.append("~ Escalated instead of warning — overly cautious")
                elif decision == ModerationAction.APPROVE:
                    under_mod_penalty = cfg["warn_clean"] * 2
                    feedback_parts.append("✗ Approved content that needed a warning")

            elif correct == ModerationAction.APPROVE:
                if decision == ModerationAction.REMOVE:
                    over_mod_penalty = cfg["remove_clean"]
                    feedback_parts.append("✗ Removed clean content — over-moderation")
                elif decision == ModerationAction.ESCALATE:
                    over_mod_penalty = cfg["remove_clean"] * 0.6
                    feedback_parts.append("✗ Escalated clean content unnecessarily")
                elif decision == ModerationAction.WARN:
                    over_mod_penalty = cfg["warn_clean"]
                    feedback_parts.append("✗ Warned clean content — minor over-moderation")

        # --- Confidence bonus/penalty ---
        confidence_bonus = 0.0
        if is_correct:
            confidence_bonus = action.confidence * cfg["confidence_bonus_scale"]
        else:
            confidence_bonus = -action.confidence * abs(cfg["confidence_penalty_scale"])

        # --- Behavioral penalty: always escalating everything ---
        consistency_penalty = 0.0
        if self._state and self._state.current_index > 3:
            total_steps = self._state.current_index
            escalate_ratio = self._state.actions_taken.get("escalate", 0) / total_steps
            if escalate_ratio > 0.8:
                consistency_penalty = cfg["escalate_all_penalty"]
                feedback_parts.append("⚠ Penalty: escalating too many posts (lazy moderation)")

            # Penalty for using same action > 85% of the time
            for act_val, count in self._state.actions_taken.items():
                if act_val == decision.value and (count / total_steps) > 0.85:
                    consistency_penalty += cfg["loop_penalty"]
                    feedback_parts.append(f"⚠ Penalty: over-relying on '{act_val}' action")
                    break

        total = accuracy_reward + confidence_bonus + consistency_penalty + over_mod_penalty + under_mod_penalty

        return Reward(
            total=round(total, 4),
            accuracy_reward=round(accuracy_reward, 4),
            confidence_bonus=round(confidence_bonus, 4),
            consistency_penalty=round(consistency_penalty, 4),
            over_moderation_penalty=round(over_mod_penalty, 4),
            under_moderation_penalty=round(under_mod_penalty, 4),
            feedback=" | ".join(feedback_parts) if feedback_parts else "No feedback",
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _calculate_episode_score(self) -> float:
        """
        Normalized score between 0.0 and 1.0 for the episode so far.
        Based on: accuracy of decisions made.
        """
        if not self._state or not self._state.decisions:
            return 0.0
        correct = sum(1 for d in self._state.decisions if d["correct"])
        return round(correct / len(self._state.decisions), 4)

    def get_final_score(self) -> float:
        """Returns final normalized episode score (0.0 to 1.0)."""
        return self._calculate_episode_score()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Returns a human-readable summary of the episode."""
        if not self._state:
            return {}
        decisions = self._state.decisions
        correct_count = sum(1 for d in decisions if d["correct"])
        return {
            "task": self.task_name,
            "total_posts": len(self._state.posts),
            "completed": self._state.current_index,
            "correct_decisions": correct_count,
            "accuracy": self.get_final_score(),
            "total_reward": round(self._state.total_reward, 4),
            "actions_distribution": self._state.actions_taken,
            "done": self._state.done,
        }
