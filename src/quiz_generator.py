"""
src/quiz_generator.py
Generates multiple-choice questions from a topic string or document text.
Used by app.py — can also be run standalone.
"""

import json
import re
import os


def generate_mcq(topic: str, num_questions: int = 5) -> list[dict]:
    """
    Generate multiple-choice questions for a given topic using Claude.

    Parameters
    ----------
    topic         : Subject name (e.g. "DSA") or raw document text.
    num_questions : How many questions to generate (3, 5, or 8 recommended).

    Returns
    -------
    List of dicts:
        [{"q": "...", "options": ["A. ...", ...], "answer": "A", "explanation": "..."}]
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    prompt = (
        f"Generate exactly {num_questions} multiple-choice questions about: {topic[:1500]}\n\n"
        "Rules:\n"
        "- 4 options each, labeled A B C D\n"
        "- One clearly correct answer\n"
        "- Include a short explanation for the correct answer\n\n"
        "Return ONLY a JSON array (no markdown, no backticks):\n"
        '[{"q":"question text","options":["A. opt","B. opt","C. opt","D. opt"],'
        '"answer":"A","explanation":"why A is correct"}]'
    )

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text
    clean = re.sub(r"```json|```", "", raw).strip()
    questions = json.loads(clean)

    # Validate structure
    for q in questions:
        assert "q" in q and "options" in q and "answer" in q, f"Malformed question: {q}"

    return questions


def print_quiz(questions: list[dict]) -> None:
    """Pretty-print a quiz to stdout for CLI use."""
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}. {q['q']}")
        for opt in q["options"]:
            marker = "✓" if opt.startswith(q["answer"]) else " "
            print(f"  [{marker}] {opt}")
        print(f"  Explanation: {q.get('explanation', '')}")


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Binary Search Trees"
    print(f"\nGenerating 5 MCQs on: {topic}\n{'='*50}")
    questions = generate_mcq(topic, num_questions=5)
    print_quiz(questions)
