"""
task_easy.py — Easy Moderation Task
=================================
Clear-cut safe vs harmful examples with highly obvious moderation outcomes.
Designed as the starter benchmark where even simple rule-based agents should
perform strongly.

Difficulty drivers:
  • Direct hate speech
  • Clear harassment
  • Obvious explicit content
  • Straightforward misinformation
  • Totally safe benign posts
  • Simple report-count cues without ambiguity
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import (
    Post,
    ViolationType,
    Severity,
    ModerationAction,
    ContentModerationEnv,
)


POSTS: list[Post] = [
    Post(
        post_id="easy_001",
        content="Good morning everyone ☀️ Hope your day is full of progress and coffee.",
        author_history="clean",
        reported_count=0,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),
    Post(
        post_id="easy_002",
        content="All immigrants are parasites and should be kicked out immediately.",
        author_history="clean",
        reported_count=12,
        platform_context="general",
        true_violation=ViolationType.HATE_SPEECH,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.REMOVE,
    ),
    Post(
        post_id="easy_003",
        content="@user You are an idiot and nobody likes you. Delete your account.",
        author_history="warned",
        reported_count=7,
        platform_context="general",
        true_violation=ViolationType.HARASSMENT,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.REMOVE,
    ),
    Post(
        post_id="easy_004",
        content="FREE giveaway!!! Click this suspicious link right now to win $5000!!!",
        author_history="warned",
        reported_count=15,
        platform_context="general",
        true_violation=ViolationType.SPAM,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.REMOVE,
    ),
    Post(
        post_id="easy_005",
        content="The Earth is flat and NASA has admitted it in leaked documents.",
        author_history="clean",
        reported_count=9,
        platform_context="news",
        true_violation=ViolationType.MISINFORMATION,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.REMOVE,
    ),
    Post(
        post_id="easy_006",
        content="Had a great workout today 💪 5km run and feeling amazing.",
        author_history="clean",
        reported_count=0,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),
    Post(
        post_id="easy_007",
        content="18+ uncensored content here 🔥 click now for private videos.",
        author_history="clean",
        reported_count=10,
        platform_context="children",
        true_violation=ViolationType.EXPLICIT,
        true_severity=Severity.CRITICAL,
        correct_action=ModerationAction.REMOVE,
    ),
    Post(
        post_id="easy_008",
        content="I baked chocolate chip cookies for the first time and they turned out perfect 🍪",
        author_history="clean",
        reported_count=0,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),
    Post(
        post_id="easy_009",
        content="People from that religion are evil and should not be trusted.",
        author_history="clean",
        reported_count=11,
        platform_context="general",
        true_violation=ViolationType.HATE_SPEECH,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.REMOVE,
    ),
    Post(
        post_id="easy_010",
        content="Breaking: drinking bleach cures viral infections instantly.",
        author_history="warned",
        reported_count=14,
        platform_context="news",
        true_violation=ViolationType.MISINFORMATION,
        true_severity=Severity.CRITICAL,
        correct_action=ModerationAction.REMOVE,
    ),
]


def make_env(seed: int = 42) -> ContentModerationEnv:
    """Return initialized easy-task environment."""
    return ContentModerationEnv(
        task_name="task_easy",
        posts=POSTS,
        seed=seed,
    )


if __name__ == "__main__":
    env = make_env(seed=0)
    obs = env.reset()

    print(f"Task: task_easy | Posts: {obs.total_steps}")
    print("-" * 60)

    from environment import Action, ModerationAction

    done = False
    step = 0
    while not done:
        action = Action(
            post_id=obs.post_id,
            decision=ModerationAction.APPROVE,
            confidence=0.6,
            reasoning="Naive approve baseline",
        )
        obs, reward, done, info = env.step(action)
        step += 1
        print(
            f"[{step:02d}] post={info['post_id']:<10} "
            f"correct={info['correct_action']:<10} "
            f"reward={reward.total:+.2f}"
        )

    print("\nSummary:", env.summary())
