"""
src/technique_checker.py
Science-backed ratings + Claude feedback for study techniques.
Used by app.py — can also be run standalone.
"""

import os

# ── Technique rating database ─────────────────────────────────────────────────
# Scores are on a 1-5 scale, based on cognitive science research.
# Sources: Dunlosky et al. (2013), Make It Stick (Brown et al.), Roediger & Karpicke (2006)

TECHNIQUE_RATINGS: dict[str, dict[str, int]] = {
    "Pomodoro": {
        "focus": 5, "retention": 4, "efficiency": 5, "science": 4,
        "notes": "Prevents mental fatigue; ideal for large tasks. Works best when break is truly restful."
    },
    "Active Recall": {
        "focus": 4, "retention": 5, "efficiency": 5, "science": 5,
        "notes": "Highest evidence-backed technique. Testing yourself beats re-reading by 2–3×."
    },
    "Spaced Repetition": {
        "focus": 3, "retention": 5, "efficiency": 5, "science": 5,
        "notes": "Optimal for long-term retention. Combine with Anki or similar tool."
    },
    "Mind Mapping": {
        "focus": 4, "retention": 4, "efficiency": 3, "science": 3,
        "notes": "Great for visual learners and concept relationships. Less effective for detail memorisation."
    },
    "Feynman Technique": {
        "focus": 5, "retention": 5, "efficiency": 4, "science": 5,
        "notes": "Teaching forces you to identify gaps. Best used after initial learning pass."
    },
    "Passive Re-reading": {
        "focus": 2, "retention": 2, "efficiency": 2, "science": 1,
        "notes": "Creates illusion of knowing. Ranked lowest in Dunlosky's 2013 meta-analysis."
    },
    "Interleaving": {
        "focus": 4, "retention": 5, "efficiency": 4, "science": 5,
        "notes": "Mixing subjects/problems within a session improves discrimination and transfer."
    },
    "Elaborative Interrogation": {
        "focus": 4, "retention": 4, "efficiency": 4, "science": 4,
        "notes": "Asking 'why' and 'how' deepens understanding beyond surface facts."
    },
}


def get_ratings(techniques: list[str]) -> dict[str, float]:
    """
    Average ratings across selected techniques.
    Unknown techniques get neutral scores (3/5).
    """
    if not techniques:
        return {"focus": 3.0, "retention": 3.0, "efficiency": 3.0, "science": 3.0}

    totals = {"focus": 0, "retention": 0, "efficiency": 0, "science": 0}
    for t in techniques:
        r = TECHNIQUE_RATINGS.get(t, {"focus": 3, "retention": 3, "efficiency": 3, "science": 3})
        for k in totals:
            totals[k] += r[k]
    n = len(techniques)
    return {k: round(v / n, 1) for k, v in totals.items()}


def analyse_technique(
    techniques: list[str],
    routine_description: str,
    subject: str = "General",
) -> str:
    """
    Get Claude's personalised feedback on a student's study approach.

    Parameters
    ----------
    techniques            : List of technique names (see TECHNIQUE_RATINGS keys).
    routine_description   : Free-text description of the student's routine.
    subject               : Subject being studied (for contextual advice).

    Returns
    -------
    Markdown-formatted feedback string from Claude.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    tech_names = techniques if techniques else ["Custom technique"]
    notes = "\n".join(
        f"- {t}: {TECHNIQUE_RATINGS[t]['notes']}"
        for t in techniques if t in TECHNIQUE_RATINGS
    )

    prompt = (
        f"A student studying {subject} uses these techniques: {', '.join(tech_names)}.\n"
        f"Their routine: \"{routine_description.strip() or 'Not described'}\"\n\n"
        f"Known research notes:\n{notes or 'N/A'}\n\n"
        "Provide personalised feedback with:\n"
        "1. **Overall Assessment** (2–3 sentences)\n"
        "2. **What's Working Well** (2–3 bullet points)\n"
        "3. **What to Improve** (3–4 specific, actionable bullet points)\n"
        "4. **One Concrete Change for This Week** (single clear recommendation)\n"
    )

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=900,
        system=(
            "You are a cognitive scientist specialising in learning efficiency. "
            "Be specific, practical, and encouraging. Avoid generic advice."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    techs    = ["Active Recall", "Spaced Repetition"]
    routine  = "I use Anki daily and quiz myself with past papers. Study 2 hrs with Pomodoro."
    subject  = "DSA"

    ratings = get_ratings(techs)
    print("\n📊 Ratings:")
    for k, v in ratings.items():
        bar = "★" * round(v) + "☆" * (5 - round(v))
        print(f"  {k.capitalize():15} {bar} ({v}/5)")

    print("\n🤖 AI Feedback:\n" + "=" * 50)
    feedback = analyse_technique(techs, routine, subject)
    print(feedback)
