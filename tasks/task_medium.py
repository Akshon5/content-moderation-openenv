"""
task_medium.py — Medium Moderation Task
========================================
Ambiguous posts, sarcasm, context-dependent decisions, and author-history
edge cases. Harder than task_easy but no adversarial phrasing or cultural
deep-dives (those are task_hard territory).

Difficulty drivers:
  • Sarcasm that reads as genuine hate if taken literally
  • Dark humour — is it a joke or a real threat?
  • Author history tips the balance (same post, different correct action)
  • Platform context changes the right call
  • Misinformation phrased as personal anecdote
  • Spam disguised as community engagement
  • Borderline political speech vs. targeted harassment
  • Low-confidence escalation cases where both warn and escalate are defensible
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

    # -------------------------------------------------------------------------
    # 1. Sarcasm — sounds like harassment, is actually mocking harassment
    # -------------------------------------------------------------------------
    Post(
        post_id="med_001",
        content=(
            "Oh sure, because CLEARLY women just aren't smart enough to code. "
            "That must be why there are so few of them in tech. Nothing to do "
            "with systemic bias at all 🙄 #Sarcasm"
        ),
        author_history="clean",
        reported_count=5,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),

    # -------------------------------------------------------------------------
    # 2. Dark humour — could be genuine threat, needs escalation not removal
    # -------------------------------------------------------------------------
    Post(
        post_id="med_002",
        content=(
            "My coworker microwaved fish AGAIN. One more time and I swear "
            "I'll flip the entire office building upside down 😤😤😤"
        ),
        author_history="clean",
        reported_count=2,
        platform_context="general",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.LOW,
        correct_action=ModerationAction.APPROVE,
    ),

    # -------------------------------------------------------------------------
    # 3. Same hyperbolic language, but repeat offender changes the calculus
    # -------------------------------------------------------------------------
    Post(
        post_id="med_003",
        content=(
            "I swear to god if my neighbour plays loud music one more night "
            "I will do something drastic. I can't take it anymore."
        ),
        author_history="repeat_offender",
        reported_count=8,
        platform_context="general",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.ESCALATE,
    ),

    # -------------------------------------------------------------------------
    # 4. Misinformation as personal anecdote — hard to disprove, still harmful
    # -------------------------------------------------------------------------
    Post(
        post_id="med_004",
        content=(
            "My doctor told me privately that masks don't actually do anything "
            "for COVID. She just can't say it publicly because of hospital policy. "
            "Thought people should know the truth 🤐"
        ),
        author_history="clean",
        reported_count=12,
        platform_context="general",
        true_violation=ViolationType.MISINFORMATION,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.WARN,
    ),

    # -------------------------------------------------------------------------
    # 5. Spam disguised as community engagement
    # -------------------------------------------------------------------------
    Post(
        post_id="med_005",
        content=(
            "Love this community so much! 💕 Everyone here is so supportive. "
            "Speaking of support, check out my friend's new supplement line — "
            "it literally cured her anxiety! Link: bit.ly/totalwellness99 🌿"
        ),
        author_history="clean",
        reported_count=3,
        platform_context="general",
        true_violation=ViolationType.SPAM,
        true_severity=Severity.LOW,
        correct_action=ModerationAction.WARN,
    ),

    # -------------------------------------------------------------------------
    # 6. Political speech — strong opinion but not targeted harassment
    # -------------------------------------------------------------------------
    Post(
        post_id="med_006",
        content=(
            "Anyone who still supports this government after what they've done "
            "to the economy is either delusional or complicit. Wake up people. 🤦"
        ),
        author_history="clean",
        reported_count=18,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),

    # -------------------------------------------------------------------------
    # 7. Same political content, but on children's platform → warn
    # -------------------------------------------------------------------------
    Post(
        post_id="med_007",
        content=(
            "Anyone who still supports this government after what they've done "
            "to the economy is either delusional or complicit. Wake up people. 🤦"
        ),
        author_history="clean",
        reported_count=4,
        platform_context="children",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.LOW,
        correct_action=ModerationAction.WARN,
    ),

    # -------------------------------------------------------------------------
    # 8. Harassment disguised as "concern" — subtly targeting a user
    # -------------------------------------------------------------------------
    Post(
        post_id="med_008",
        content=(
            "I'm just genuinely worried about @sarah_dev. She posts so much "
            "about her startup failing. Is she okay? Does anyone know where "
            "she lives so we can check on her? Asking for her wellbeing 😊"
        ),
        author_history="warned",
        reported_count=9,
        platform_context="general",
        true_violation=ViolationType.HARASSMENT,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 9. Borderline health misinformation — anecdotal, not dangerous enough to remove
    # -------------------------------------------------------------------------
    Post(
        post_id="med_009",
        content=(
            "Stopped taking my antidepressants cold turkey 3 weeks ago and "
            "honestly feel AMAZING. Sometimes you just need to trust your body "
            "and ditch the pharma 💊❌ Anyone else done this?"
        ),
        author_history="clean",
        reported_count=6,
        platform_context="general",
        true_violation=ViolationType.MISINFORMATION,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.WARN,
    ),

    # -------------------------------------------------------------------------
    # 10. Genuine product review that sounds like spam
    # -------------------------------------------------------------------------
    Post(
        post_id="med_010",
        content=(
            "Honestly blown away by the new Dyson vacuum. Picked it up last week "
            "and it's genuinely the best purchase I've made all year. "
            "No affiliation, just a happy customer sharing my experience!"
        ),
        author_history="clean",
        reported_count=0,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),

    # -------------------------------------------------------------------------
    # 11. Escalate — suicide-adjacent content, ambiguous intent
    # -------------------------------------------------------------------------
    Post(
        post_id="med_011",
        content=(
            "Been thinking a lot lately about how much easier things would be "
            "if I just wasn't here anymore. Not saying I'd do anything. "
            "Just tired, you know?"
        ),
        author_history="clean",
        reported_count=3,
        platform_context="general",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.ESCALATE,
    ),

    # -------------------------------------------------------------------------
    # 12. Repeat spammer — same low-level spam, now warrants escalation
    # -------------------------------------------------------------------------
    Post(
        post_id="med_012",
        content=(
            "Hey guys! Just sharing another amazing opportunity I found online. "
            "Work from home, earn $500/day, no experience needed! DM me for details!"
        ),
        author_history="repeat_offender",
        reported_count=20,
        platform_context="general",
        true_violation=ViolationType.SPAM,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 13. Ambiguous callout post — could be holding someone accountable or harassment
    # -------------------------------------------------------------------------
    Post(
        post_id="med_013",
        content=(
            "PSA: @techbro_mike is a known scammer who took money from 12 people "
            "in our community and never delivered. Screenshots available. "
            "Protect yourselves. 🚨"
        ),
        author_history="clean",
        reported_count=7,
        platform_context="general",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.ESCALATE,
    ),

    # -------------------------------------------------------------------------
    # 14. Edgy humour — offensive but falls within adult platform rules
    # -------------------------------------------------------------------------
    Post(
        post_id="med_014",
        content=(
            "Why do programmers always mix up Halloween and Christmas? "
            "Because Oct 31 = Dec 25. 💀 Nerd jokes are the only jokes "
            "that get darker the more you understand them."
        ),
        author_history="clean",
        reported_count=0,
        platform_context="adult",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),

    # -------------------------------------------------------------------------
    # 15. Coordinated complaint about a business — not harassment, just review
    # -------------------------------------------------------------------------
    Post(
        post_id="med_015",
        content=(
            "DO NOT use ServiceCo. They charged me twice, ignored my emails "
            "for 3 weeks, and hung up on me when I called. Absolute scam. "
            "1 star. #ConsumerWarning"
        ),
        author_history="clean",
        reported_count=2,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),

    # -------------------------------------------------------------------------
    # 16. Warn — body-shaming disguised as fitness motivation
    # -------------------------------------------------------------------------
    Post(
        post_id="med_016",
        content=(
            "No excuses. If you're not hitting the gym 5x a week you're just "
            "choosing to be unhealthy and lazy. Discipline > feelings. "
            "Summer bodies are made in winter 💪 #FitnessMotivation"
        ),
        author_history="clean",
        reported_count=11,
        platform_context="general",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.LOW,
        correct_action=ModerationAction.WARN,
    ),

    # -------------------------------------------------------------------------
    # 17. News platform — speculation about ongoing legal case (escalate)
    # -------------------------------------------------------------------------
    Post(
        post_id="med_017",
        content=(
            "My source inside the courthouse says the defendant is definitely "
            "going to be found guilty. The jury has already made up their minds. "
            "Justice incoming. #BreakingNews"
        ),
        author_history="clean",
        reported_count=14,
        platform_context="news",
        true_violation=ViolationType.MISINFORMATION,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.ESCALATE,
    ),

    # -------------------------------------------------------------------------
    # 18. Ragebait title with perfectly fine content
    # -------------------------------------------------------------------------
    Post(
        post_id="med_018",
        content=(
            "UNPOPULAR OPINION: pineapple DOES belong on pizza and I will die "
            "on this hill 🍕🍍 Fight me in the comments. This is the hill I choose."
        ),
        author_history="clean",
        reported_count=30,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),

    # -------------------------------------------------------------------------
    # 19. Passive-aggressive targeted post — borderline harassment, warn
    # -------------------------------------------------------------------------
    Post(
        post_id="med_019",
        content=(
            "Funny how some people in this community think they can just "
            "talk over everyone else and take credit for others' work. "
            "You know who you are. Some of us have noticed. Just saying. 😶"
        ),
        author_history="warned",
        reported_count=5,
        platform_context="general",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.LOW,
        correct_action=ModerationAction.WARN,
    ),

    # -------------------------------------------------------------------------
    # 20. Genuine mental health advocacy — looks alarming but is healthy
    # -------------------------------------------------------------------------
    Post(
        post_id="med_020",
        content=(
            "3 years ago I was at my lowest. Today I'm sharing my story because "
            "mental health struggles are real and recovery is possible. "
            "If you're struggling, please reach out — you're not alone 💙 "
            "#MentalHealthAwareness"
        ),
        author_history="clean",
        reported_count=1,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),
]


def make_env(seed: int = 42) -> ContentModerationEnv:
    """
    Returns a fully initialized medium-task environment.

    Args:
        seed: Random seed for post shuffling reproducibility.

    Returns:
        ContentModerationEnv ready to call reset() / step() on.
    """
    return ContentModerationEnv(
        task_name="task_medium",
        posts=POSTS,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Quick sanity-check runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = make_env(seed=0)
    obs = env.reset()

    print(f"Task: task_medium | Posts: {obs.total_steps}")
    print("-" * 60)

    step = 0
    while True:
        from environment import Action, ModerationAction
        action = Action(
            post_id=obs.post_id,
            decision=ModerationAction.APPROVE,
            confidence=0.5,
            reasoning="Baseline: always approve",
        )
        obs, reward, done, info = env.step(action)
        step += 1
        print(
            f"[{step:02d}] post={info['post_id']:<12} "
            f"correct={info['correct_action']:<10} "
            f"agent=approve    "
            f"reward={reward.total:+.3f}  "
            f"feedback={reward.feedback[:60]}"
        )
        if done:
            break

    summary = env.summary()
    print("\n" + "=" * 60)
    print(f"Final Score : {summary['accuracy']:.2%}")
    print(f"Total Reward: {summary['total_reward']:.4f}")
    print(f"Actions     : {summary['actions_distribution']}")
