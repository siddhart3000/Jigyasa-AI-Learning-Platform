from __future__ import annotations

import pandas as pd

from .storage import load_all_students_raw


def score_row(row: dict) -> int:
    # Points system:
    # - Videos: 5 each
    # - Quizzes attempted: 10 each
    # - AI questions: 2 each
    # - Lessons completed: 15 each
    return (
        int(row.get("videos_watched", 0)) * 5
        + int(row.get("quizzes_attempted", 0)) * 10
        + int(row.get("ai_questions_asked", 0)) * 2
        + int(row.get("lessons_completed", 0)) * 15
        + int(row.get("xp", 0))
    )


def learning_score_from_row(row: dict) -> float:
    quiz_total = int(row.get("quiz_total", 0) or 0)
    quiz_correct = int(row.get("quiz_correct", 0) or 0)
    accuracy = 0.0 if quiz_total <= 0 else (quiz_correct / max(1, quiz_total)) * 100.0

    study_time_minutes = int(row.get("study_time_minutes", 0) or 0)
    practice_attempts = int(row.get("practice_attempts", 0) or 0)
    ai_questions = int(row.get("ai_questions_asked", 0) or 0)

    time_spent = min(100.0, (study_time_minutes / 300.0) * 100.0)
    practice = min(100.0, (practice_attempts / 50.0) * 100.0)
    ai_use = min(100.0, (ai_questions / 40.0) * 100.0)
    return 0.4 * accuracy + 0.3 * time_spent + 0.2 * practice + 0.1 * ai_use


def leaderboard_df() -> pd.DataFrame:
    students = load_all_students_raw()
    if not students:
        return pd.DataFrame(columns=["Rank", "Name", "Email", "Score", "XP"])

    rows = []
    for s in students:
        score = score_row(s)
        lscore = learning_score_from_row(s)
        rows.append(
            {
                "Name": s.get("name", "Student"),
                "Email": s.get("email", ""),
                "Role": s.get("role", "Student"),
                "Score": score,
                "XP": int(s.get("xp", 0)),
                "Learning Score": round(lscore, 2),
                "Videos": int(s.get("videos_watched", 0)),
                "Quizzes": int(s.get("quizzes_attempted", 0)),
                "AI Qs": int(s.get("ai_questions_asked", 0)),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(["Score", "Learning Score", "XP"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    df.insert(0, "Rank", df.index + 1)
    return df
