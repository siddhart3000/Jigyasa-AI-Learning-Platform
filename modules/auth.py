from __future__ import annotations
import bcrypt
from datetime import datetime
from dataclasses import asdict
from modules.firebase_service import db
from modules.storage import StudentState

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False

def signup_user(name: str, email: str, password: str) -> tuple[bool, str]:
    email = email.lower().strip()
    try:
        user_ref = db.collection("users").document(email)
        if user_ref.get().exists:
            return False, "Email already registered."
        
        new_student = StudentState(
            name=name,
            email=email,
            role="student",
            password_hash=hash_password(password),
            created_at=datetime.now().isoformat(),
            xp=0,
            quizzes_attempted=0,
            ai_questions_asked=0,
            videos_watched=0,
            study_time_minutes=0
        )
        
        user_ref.set(asdict(new_student))
        return True, "Account created successfully!"
    except Exception as e:
        return False, f"Signup error: {str(e)}"

def login_user(email: str, password: str) -> tuple[bool, str | StudentState]:
    email = email.lower().strip()
    try:
        user_doc = db.collection("users").document(email).get()
        if not user_doc.exists:
            return False, "User not found."
        
        data = user_doc.to_dict()
        stored_hash = data.get("password_hash", "")
        
        if not stored_hash:
            return False, "Account error: No password set."
            
        if check_password(password, stored_hash):
            return True, StudentState(**data)
        else:
            return False, "Invalid password."
    except Exception as e:
        return False, f"Login error: {str(e)}"