from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def simulate(n: int, seed: int = 42) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    # =========================================================
    # STUDENT IDS
    # =========================================================

    student_id = [f"STU_{1000 + i}" for i in range(n)]

    # =========================================================
    # STUDENT TYPES
    # =========================================================

    student_type = rng.choice(
        ["consistent", "lazy_smart", "hardworking", "weak"],
        size=n,
        p=[0.40, 0.20, 0.25, 0.15]
    )

    # =========================================================
    # BASE FEATURES
    # =========================================================

    study_time = rng.integers(30, 360, size=n)
    quiz_attempts = rng.integers(0, 30, size=n)

    quiz_accuracy = np.clip(
        rng.normal(65, 18, size=n),
        0,
        100
    )

    ai_questions = rng.integers(0, 60, size=n)

    videos_watched = rng.integers(0, 25, size=n)

    practice_attempts = rng.integers(0, 50, size=n)

    # =========================================================
    # STUDENT TYPE BEHAVIOR
    # =========================================================

    # Lazy but intelligent students
    lazy_mask = student_type == "lazy_smart"

    quiz_accuracy = np.where(
        lazy_mask,
        quiz_accuracy + rng.normal(10, 5, size=n),
        quiz_accuracy
    )

    study_time = np.where(
        lazy_mask,
        study_time - rng.integers(10, 60, size=n),
        study_time
    )

    # Hardworking students
    hard_mask = student_type == "hardworking"

    study_time = np.where(
        hard_mask,
        study_time + rng.integers(40, 120, size=n),
        study_time
    )

    practice_attempts = np.where(
        hard_mask,
        practice_attempts + rng.integers(10, 25, size=n),
        practice_attempts
    )

    # Weak students
    weak_mask = student_type == "weak"

    quiz_accuracy = np.where(
        weak_mask,
        quiz_accuracy - rng.normal(12, 6, size=n),
        quiz_accuracy
    )

    # =========================================================
    # OUTLIERS
    # =========================================================

    outlier_idx = rng.choice(
        n,
        size=max(1, n // 20),
        replace=False
    )

    study_time[outlier_idx] *= 2
    ai_questions[outlier_idx] *= 2

    # =========================================================
    # CLIP VALUES
    # =========================================================

    study_time = np.clip(study_time, 0, 600)

    quiz_accuracy = np.clip(
        quiz_accuracy,
        0,
        100
    )

    ai_questions = np.clip(
        ai_questions,
        0,
        120
    )

    practice_attempts = np.clip(
        practice_attempts,
        0,
        80
    )

    # =========================================================
    # ENGAGEMENT SCORE
    # =========================================================

    engagement_score = np.clip(
        (
            0.30 * np.clip(study_time / 300 * 100, 0, 100)
            + 0.25 * np.clip(quiz_attempts / 20 * 100, 0, 100)
            + 0.25 * np.clip(ai_questions / 50 * 100, 0, 100)
            + 0.20 * np.clip(videos_watched / 30 * 100, 0, 100)
        )
        + rng.normal(0, 5, size=n),
        0,
        100
    )

    # =========================================================
    # TOPIC MASTERY
    # =========================================================

    topic_mastery = np.clip(
        (
            0.35 * quiz_accuracy
            + 0.35 * np.clip(practice_attempts / 50 * 100, 0, 100)
            + 0.30 * np.clip(study_time / 360 * 100, 0, 100)
        )
        + rng.normal(0, 8, size=n),
        0,
        100
    )

    # =========================================================
    # STUDY CONSISTENCY
    # =========================================================

    study_consistency = np.clip(
        (
            0.5 * np.clip(study_time / 360 * 100, 0, 100)
            + 0.5 * np.clip(quiz_attempts / 25 * 100, 0, 100)
        )
        + rng.normal(0, 10, size=n),
        0,
        100
    )

    # =========================================================
    # PREDICTED EXAM SCORE
    # =========================================================

    exam_score = np.clip(
        (
            0.40 * quiz_accuracy
            + 0.25 * topic_mastery
            + 0.20 * engagement_score
            + 0.15 * study_consistency
        )
        + rng.normal(0, 6, size=n),
        0,
        100
    )

    # =========================================================
    # MISSING VALUES FOR REALISM
    # =========================================================

    missing_mask = rng.random(n) < 0.05

    quiz_accuracy = quiz_accuracy.astype(float)

    quiz_accuracy[missing_mask] = np.nan

    # =========================================================
    # FINAL DATAFRAME
    # =========================================================

    df = pd.DataFrame(
        {
            "student_id": student_id,
            "student_type": student_type,
            "study_time_minutes": study_time,
            "quiz_attempts": quiz_attempts,
            "quiz_accuracy": np.round(quiz_accuracy, 2),
            "ai_questions_asked": ai_questions,
            "videos_watched": videos_watched,
            "practice_attempts": practice_attempts,
            "engagement_score": np.round(engagement_score, 2),
            "topic_mastery": np.round(topic_mastery, 2),
            "study_consistency": np.round(study_consistency, 2),
            "exam_score": np.round(exam_score, 2),
        }
    )

    return df


def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--n",
        type=int,
        default=500
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    ap.add_argument(
        "--out",
        type=str,
        default="research/student_learning_data.csv"
    )

    args = ap.parse_args()

    df = simulate(
        args.n,
        args.seed
    )

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        out,
        index=False
    )

    print(f"\nGenerated {len(df)} student records")
    print(f"Saved dataset to: {out.resolve()}")
    print("\nDataset Preview:\n")
    print(df.head())


if __name__ == "__main__":
    main()