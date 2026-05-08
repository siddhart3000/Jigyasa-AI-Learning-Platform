from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from datetime import date
from modules.firebase_service import db

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
APP_STATE_PATH = DATA_DIR / "app_state.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class StudentState:
    name: str = "Student"
    email: str = "student@example.com"
    role: str = "Student"
    password_hash: str = ""
    created_at: str = ""
    xp: int = 0
    lessons_completed: int = 0
    quizzes_attempted: int = 0
    quiz_correct: int = 0
    quiz_total: int = 0
    videos_watched: int = 0
    video_watch_count: int = 0
    video_watch_time_minutes: int = 0
    ai_questions_asked: int = 0
    study_time_minutes: int = 0
    practice_attempts: int = 0
    summary_requests: int = 0
    topic_quiz_performance: dict[str, dict[str, int]] = None  # type: ignore[assignment]
    favorite_summaries: list[dict[str, Any]] = None  # type: ignore[assignment]
    daily_activity: dict[str, dict[str, int]] = None  # type: ignore[assignment]

    watched_video_ids: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.watched_video_ids is None:
            self.watched_video_ids = []
        if self.daily_activity is None:
            self.daily_activity = {}
        if self.topic_quiz_performance is None:
            self.topic_quiz_performance = {}
        if self.favorite_summaries is None:
            self.favorite_summaries = []


def load_student(email: str) -> StudentState:
    try:
        doc = db.collection("users").document(email).get()
        if doc.exists:
            raw = doc.to_dict()
            return StudentState(**raw)
    except Exception:
        pass
    return StudentState(email=email)


def save_student(student: StudentState) -> None:
    db.collection("users").document(student.email).set(asdict(student))


def upsert_student_profile(email: str, name: str) -> StudentState:
    student = load_student(email)
    student.email = email
    # Only update name if it's default or new
    student.name = name
    save_student(student)
    return student


def load_all_students_raw() -> list[dict[str, Any]]:
    docs = db.collection("users").stream()
    return [doc.to_dict() for doc in docs if not doc.id.startswith("_")]


def log_daily_activity(student: StudentState, key: str, amount: int = 1) -> None:
    today = date.today().isoformat()
    bucket = student.daily_activity.setdefault(today, {})
    bucket[key] = int(bucket.get(key, 0)) + int(amount)
