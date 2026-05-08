from __future__ import annotations

import pandas as pd

from .storage import StudentState


# =========================================================
# QUIZ ACCURACY
# =========================================================
def quiz_accuracy(student: StudentState) -> float:
    """
    Calculates overall quiz accuracy percentage.
    """

    if getattr(student, "quiz_total", 0) <= 0:
        return 0.0

    accuracy = (
        student.quiz_correct / max(1, student.quiz_total)
    ) * 100.0

    return round(max(0.0, min(100.0, accuracy)), 2)


# =========================================================
# LEARNING SCORE
# =========================================================
def learning_score(student: StudentState) -> float:
    """
    Overall AI learning intelligence score.
    """

    accuracy = quiz_accuracy(student)

    time_spent = min(
        100.0,
        (student.study_time_minutes / 300.0) * 100.0
    )

    practice = min(
        100.0,
        (student.practice_attempts / 50.0) * 100.0
    )

    ai_use = min(
        100.0,
        (student.ai_questions_asked / 40.0) * 100.0
    )

    summary_use = min(
        100.0,
        (student.summary_requests / 30.0) * 100.0
    )

    score = (
        0.30 * accuracy
        + 0.25 * time_spent
        + 0.20 * practice
        + 0.15 * ai_use
        + 0.10 * summary_use
    )

    return round(max(0.0, min(100.0, score)), 2)


# =========================================================
# ENGAGEMENT SCORE
# =========================================================
def engagement_score(student: StudentState) -> float:
    """
    Measures how actively the student uses the platform.
    """

    study = min(
        100.0,
        (student.study_time_minutes / 300.0) * 100.0
    )

    quiz = min(
        100.0,
        (student.quizzes_attempted / 20.0) * 100.0
    )

    aiq = min(
        100.0,
        (student.ai_questions_asked / 50.0) * 100.0
    )

    vids = min(
        100.0,
        (student.videos_watched / 30.0) * 100.0
    )

    score = (
        0.30 * study
        + 0.25 * quiz
        + 0.25 * aiq
        + 0.20 * vids
    )

    return round(max(0.0, min(100.0, score)), 2)


# =========================================================
# TOPIC MASTERY
# =========================================================
def topic_mastery(student: StudentState) -> dict[str, float]:
    """
    Estimates mastery per subject/topic.
    """

    base_accuracy = quiz_accuracy(student)

    scores = {
        "Math": min(
            100.0,
            base_accuracy + min(20.0, student.videos_watched * 1.5)
        ),

        "Science": min(
            100.0,
            base_accuracy + min(15.0, student.ai_questions_asked * 0.6)
        ),

        "English": min(
            100.0,
            base_accuracy + min(
                10.0,
                student.study_time_minutes / 60.0
            )
        ),
    }

    # Real per-topic performance override
    for topic, perf in (
        student.topic_quiz_performance or {}
    ).items():

        total = int(perf.get("total", 0))
        correct = int(perf.get("correct", 0))

        if total <= 0:
            continue

        scores[topic] = round(
            (correct / total) * 100.0,
            2
        )

    return scores


# =========================================================
# LEARNING PROFILE DETECTION
# =========================================================
def detect_learning_profile(student: StudentState) -> tuple[str, str, str]:
    """
    Classifies student into behavioral profiles.
    Returns (Profile Name, Explanation, Recommendation).
    """
    qa = quiz_accuracy(student)
    cons = study_consistency_score(student)
    study_time = student.study_time_minutes
    ai_qs = student.ai_questions_asked
    practice = student.practice_attempts

    if qa > 80 and study_time < 120 and cons > 60:
        return (
            "Smart Performer",
            "You grasp concepts quickly with high accuracy and efficient study time.",
            "Try more 'Hard' difficulty quizzes to push your limits."
        )
    
    if cons > 80:
        return (
            "Consistent Learner",
            "Your daily study habit is excellent. Consistency is your greatest strength.",
            "Keep up the streak! Focus on deep-diving into complex topics using the AI Tutor."
        )

    if study_time > 180 or practice > 30:
        return (
            "Hardworking Improver",
            "You are putting in significant effort and practice hours.",
            "Review your incorrect quiz answers to turn that hard work into higher accuracy."
        )

    if qa < 50 or (cons < 40 and study_time < 60):
        return (
            "At-Risk Student",
            "Current engagement and accuracy levels suggest you might be struggling.",
            "Start with 'Easy' quizzes and watch 2-3 conceptual videos today."
        )

    return (
        "Steady Progressor",
        "You are maintaining a balanced learning pace.",
        "Try to increase your daily study time by 15 minutes to reach the next level."
    )


# =========================================================
# LEARNING HEALTH SCORE
# =========================================================
def learning_health_score(student: StudentState) -> float:
    qa = quiz_accuracy(student)
    cons = study_consistency_score(student)
    eng = engagement_score(student)
    study_time_norm = min(100.0, (student.study_time_minutes / 360.0) * 100.0)
    
    score = (0.3 * qa) + (0.3 * cons) + (0.2 * eng) + (0.2 * study_time_norm)
    return round(score, 2)

# =========================================================
# STUDY CONSISTENCY
# =========================================================
def study_consistency_score(
    student: StudentState,
    window_days: int = 7
) -> float:
    """
    Measures how consistently the student studies.
    """

    if not student.daily_activity:
        return 0.0

    days = sorted(
        student.daily_activity.keys()
    )[-window_days:]

    if not days:
        return 0.0

    active_days = 0

    for d in days:

        bucket = student.daily_activity.get(d, {})

        total_activity = (
            bucket.get("study_minutes", 0)
            + bucket.get("video_minutes", 0)
            + bucket.get("ai_questions", 0)
            + bucket.get("quiz_attempts", 0)
        )

        if total_activity > 0:
            active_days += 1

    consistency = (
        active_days / max(1, len(days))
    ) * 100.0

    return round(consistency, 2)


# =========================================================
# EXAM PREDICTION
# =========================================================
def predict_exam_score(student: StudentState) -> float:
    """
    AI-based predicted exam score.
    """

    qa = quiz_accuracy(student)

    mastery_scores = list(
        topic_mastery(student).values()
    )

    if mastery_scores:
        mastery = (
            sum(mastery_scores)
            / len(mastery_scores)
        )
    else:
        mastery = 0.0

    engage = engagement_score(student)

    study_time_score = min(
        100.0,
        (student.study_time_minutes / 300.0) * 100.0
    )

    predicted = (
        0.45 * qa
        + 0.25 * mastery
        + 0.15 * engage
        + 0.15 * study_time_score
    )

    return round(
        max(0.0, min(100.0, predicted)),
        2
    )

def exam_performance_band(score: float) -> str:
    """
    Converts predicted score to professional bands.
    """
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Average"
    if score >= 35:
        return "Needs Improvement"
    return "Critical Attention Required"


# =========================================================
# ANALYTICS DATAFRAME
# =========================================================
def build_analytics_frame(
    student: StudentState
) -> pd.DataFrame:
    """
    Creates dataframe for charts/graphs.
    """

    return pd.DataFrame(
        [
            {
                "Metric": "Quiz Accuracy (%)",
                "Value": quiz_accuracy(student),
            },
            {
                "Metric": "Study Time (min)",
                "Value": student.study_time_minutes,
            },
            {
                "Metric": "Practice Attempts",
                "Value": student.practice_attempts,
            },
            {
                "Metric": "AI Questions",
                "Value": student.ai_questions_asked,
            },
            {
                "Metric": "Summary Usage",
                "Value": student.summary_requests,
            },
            {
                "Metric": "Videos Watched",
                "Value": student.videos_watched,
            },
            {
                "Metric": "Engagement Score",
                "Value": engagement_score(student),
            },
            {
                "Metric": "Learning Score",
                "Value": learning_score(student),
            },
            {
                "Metric": "Consistency Score",
                "Value": study_consistency_score(student),
            },
            {
                "Metric": "Predicted Exam Score",
                "Value": predict_exam_score(student),
            },
        ]
    )


# =========================================================
# PERFORMANCE BAND
# =========================================================
def detect_strength_band(score: float) -> str:

    if score < 40:
        return "Weak (needs practice)"

    if score < 70:
        return "Average (improving)"

    if score < 85:
        return "Good"

    return "Excellent"