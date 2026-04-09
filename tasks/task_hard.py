"""
task_hard.py — Hard Moderation Task
=====================================
Multi-rule violations, cultural nuance, adversarial phrasing, platform-context
traps, and author-history edge cases. Designed to challenge agents that rely on
surface-level pattern matching.

Difficulty drivers:
  • Coded language / dog-whistles that look innocuous in isolation
  • Satire / parody that could be genuine hate speech
  • Misinformation phrased as "just asking questions"
  • Clean posts by repeat offenders (over-moderation trap)
  • Hate speech using academic or clinical framing
  • Context-flip posts (fine on adult platform, violation on children's)
  • Multi-rule: simultaneous hate + misinformation, spam + harassment, etc.
  • Adversarial: deliberately broken spelling, homoglyphs, zero-width chars
  • High report counts on clean posts (brigading simulation)
  • Low report counts on severe violations (coordinated bad actors)
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
    # 1. Coded hate speech — "14 words" dog-whistle, looks like a life motto
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_001",
        content=(
            "We must secure the existence of our people and a future for our children. "
            "Family first, always. 🌿"
        ),
        author_history="clean",
        reported_count=2,
        platform_context="general",
        true_violation=ViolationType.HATE_SPEECH,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 2. Satire account — obvious parody of a politician, but reads as real threat
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_002",
        content=(
            "PARODY ACCOUNT 😂 — 'I hereby declare that anyone who votes against me "
            "is a traitor and deserves what's coming to them.' [This is satire, relax]"
        ),
        author_history="clean",
        reported_count=45,
        platform_context="news",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.ESCALATE,
    ),

    # -------------------------------------------------------------------------
    # 3. Misinformation as "just asking questions" — vaccine disinfo
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_003",
        content=(
            "Not saying vaccines caused my cousin's autism, I'm just saying the timing "
            "is suspicious. Has anyone else noticed this? Would love to hear your "
            "experiences 🤔 #JustCurious #DoYourResearch"
        ),
        author_history="warned",
        reported_count=8,
        platform_context="general",
        true_violation=ViolationType.MISINFORMATION,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 4. Repeat offender posts genuinely clean content (over-moderation trap)
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_004",
        content=(
            "Finally finished my sourdough starter after 7 days of feeding it. "
            "First loaf came out perfectly! So happy 🍞"
        ),
        author_history="repeat_offender",
        reported_count=0,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),

    # -------------------------------------------------------------------------
    # 5. Adversarial spelling — h4te speech with l33tspeak / homoglyphs
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_005",
        content=(
            "All [slur spelled: k1ll 4ll imm1grants] should be d3ported immediately. "
            "Our country belongs to real citizens only. 🇺🇸🇺🇸🇺🇸"
        ),
        author_history="warned",
        reported_count=3,
        platform_context="general",
        true_violation=ViolationType.HATE_SPEECH,
        true_severity=Severity.CRITICAL,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 6. Multi-rule: harassment + doxxing threat (no actual PII, but threatening)
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_006",
        content=(
            "@journalist_jane you think you're safe? We know where you live. "
            "Your neighbors know what you write. Accidents happen to people like you."
        ),
        author_history="clean",
        reported_count=12,
        platform_context="general",
        true_violation=ViolationType.HARASSMENT,
        true_severity=Severity.CRITICAL,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 7. Cultural nuance — phrase is a slur in one culture, a compliment in another
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_007",
        content=(
            "My nonna always called me her little 'terrone' with so much love. "
            "Regional Italian identity is something to be proud of! 🇮🇹"
        ),
        author_history="clean",
        reported_count=6,
        platform_context="general",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.LOW,
        correct_action=ModerationAction.ESCALATE,
    ),

    # -------------------------------------------------------------------------
    # 8. Context flip — explicit but legal adult content on children's platform
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_008",
        content=(
            "Check out my new story where two adults explore their desires 🔥 "
            "Link in bio — adults only, you've been warned 😉"
        ),
        author_history="clean",
        reported_count=22,
        platform_context="children",
        true_violation=ViolationType.EXPLICIT,
        true_severity=Severity.CRITICAL,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 9. Academic framing of hate — using pseudoscience to justify racism
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_009",
        content=(
            "Multiple peer-reviewed studies (see citations below) show measurable "
            "IQ differences between racial groups. This is just biology, not racism. "
            "Why is science being censored? #FreeSpeech"
        ),
        author_history="clean",
        reported_count=5,
        platform_context="general",
        true_violation=ViolationType.HATE_SPEECH,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 10. Brigaded clean post — 200 reports but content is totally fine
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_010",
        content=(
            "Proud to announce I'm voting for [opposition party] in the upcoming election. "
            "Here's why their healthcare policy makes sense to me: [thoughtful 3-point argument]"
        ),
        author_history="clean",
        reported_count=200,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),

    # -------------------------------------------------------------------------
    # 11. Coordinated bad actor — severe violation, 0 reports (coordinated silence)
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_011",
        content=(
            "Step-by-step: how to acquire unregistered firearms at weekend markets. "
            "DM me for the full guide. These are just 'tools' 😉"
        ),
        author_history="clean",
        reported_count=0,
        platform_context="general",
        true_violation=ViolationType.EXPLICIT,   # Policy: illegal activity facilitation
        true_severity=Severity.CRITICAL,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 12. Spam + harassment hybrid — fake giveaway targeting a specific user
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_012",
        content=(
            "🎁 FREE iPhone giveaway! Just RT and follow! Also @targetuser you are "
            "a disgusting excuse for a human and I hope your business fails. "
            "Enter the giveaway now!!! Limited time!!!"
        ),
        author_history="warned",
        reported_count=15,
        platform_context="general",
        true_violation=ViolationType.HARASSMENT,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 13. Self-harm coded language — appears to be wellness advice
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_013",
        content=(
            "When the pain gets too much, there's always a permanent solution to a "
            "temporary problem. Sometimes the kindest thing you can do is let go entirely. "
            "Sending love to those who understand 💙"
        ),
        author_history="clean",
        reported_count=4,
        platform_context="general",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.ESCALATE,
    ),

    # -------------------------------------------------------------------------
    # 14. Misinformation + hate combo — COVID origin theory with xenophobia
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_014",
        content=(
            "The Chinese government deliberately engineered and released COVID-19 "
            "as a bioweapon against the West. All Chinese people in our country "
            "should be considered security risks. This is just facts people. 🦠"
        ),
        author_history="repeat_offender",
        reported_count=30,
        platform_context="news",
        true_violation=ViolationType.MISINFORMATION,
        true_severity=Severity.CRITICAL,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 15. Repeat offender, borderline content — warn not remove
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_015",
        content=(
            "Honestly these new diversity hiring quotas are reverse discrimination. "
            "Merit should be the only metric. People are getting jobs they don't deserve."
        ),
        author_history="repeat_offender",
        reported_count=9,
        platform_context="general",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.WARN,
    ),

    # -------------------------------------------------------------------------
    # 16. Zero-width character obfuscation — hate hidden in invisible chars
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_016",
        content=(
            "I love all people of every background\u200b\u200b\u200b "
            "[hidden: except [group], they should be eliminated] "
            "— unity is our strength 🌍"
        ),
        author_history="clean",
        reported_count=1,
        platform_context="general",
        true_violation=ViolationType.HATE_SPEECH,
        true_severity=Severity.CRITICAL,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 17. Legitimate medical advice that sounds like drug promotion
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_017",
        content=(
            "As a nurse: if you're experiencing opioid withdrawal, low-dose methadone "
            "or buprenorphine under physician supervision is highly effective. "
            "Please seek a licensed addiction specialist — this is a medical emergency."
        ),
        author_history="clean",
        reported_count=7,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),

    # -------------------------------------------------------------------------
    # 18. Escalation trap — appears critical but lacks enough evidence to remove
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_018",
        content=(
            "I know exactly what [politician name] has been doing behind closed doors. "
            "The evidence will come out soon. Everyone will be shocked. Stay tuned. 👀"
        ),
        author_history="warned",
        reported_count=18,
        platform_context="news",
        true_violation=ViolationType.BORDERLINE,
        true_severity=Severity.MEDIUM,
        correct_action=ModerationAction.ESCALATE,
    ),

    # -------------------------------------------------------------------------
    # 19. Hate speech disguised as comedy / meme format
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_019",
        content=(
            "POV: You're a [ethnic group] trying to do math 💀💀💀 "
            "Just kidding (not really) 😂 #Comedy #JustAJoke #NoOffense"
        ),
        author_history="clean",
        reported_count=14,
        platform_context="general",
        true_violation=ViolationType.HATE_SPEECH,
        true_severity=Severity.HIGH,
        correct_action=ModerationAction.REMOVE,
    ),

    # -------------------------------------------------------------------------
    # 20. Genuine political criticism — looks like harassment but is not
    # -------------------------------------------------------------------------
    Post(
        post_id="hard_020",
        content=(
            "Senator [Name]'s vote against the climate bill was a moral failure "
            "and a betrayal of her constituents. She should resign. "
            "#ClimateAction #AccountabilityNow"
        ),
        author_history="clean",
        reported_count=3,
        platform_context="general",
        true_violation=ViolationType.NONE,
        true_severity=Severity.NONE,
        correct_action=ModerationAction.APPROVE,
    ),
]


def make_env(seed: int = 42) -> ContentModerationEnv:
    """
    Returns a fully initialized hard-task environment.

    Args:
        seed: Random seed for post shuffling reproducibility.

    Returns:
        ContentModerationEnv ready to call reset() / step() on.
    """
    return ContentModerationEnv(
        task_name="task_hard",
        posts=POSTS,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Quick sanity-check runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = make_env(seed=0)
    obs = env.reset()

    print(f"Task: task_hard | Posts: {obs.total_steps}")
    print("-" * 60)

    step = 0
    while True:
        # Naive baseline: always escalate (should score poorly due to behavioral penalty)
        from environment import Action, ModerationAction
        action = Action(
            post_id=obs.post_id,
            decision=ModerationAction.ESCALATE,
            confidence=0.5,
            reasoning="Baseline: always escalate",
        )
        obs, reward, done, info = env.step(action)
        step += 1
        print(
            f"[{step:02d}] post={info['post_id']:<12} "
            f"correct={info['correct_action']:<10} "
            f"agent=escalate  "
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
