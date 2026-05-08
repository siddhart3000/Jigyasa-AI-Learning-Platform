from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Jigyasa AI Service")

# Make root project modules importable (../modules, ../pdf_library, etc.)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional imports (fail gracefully with helpful errors).
try:
    from modules.ai_tutor import tutor_answer, answer_question, generate_summary, generate_key_points
    from modules.pdf_reader import build_context
    from modules.quiz_generator import QuizRequest, generate_pdf_quiz
except Exception as e:  # pragma: no cover
    tutor_answer = None  # type: ignore
    answer_question = None  # type: ignore
    generate_summary = None  # type: ignore
    generate_key_points = None  # type: ignore
    build_context = None  # type: ignore
    QuizRequest = None  # type: ignore
    generate_pdf_quiz = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    lesson: dict | None = None

class LessonRef(BaseModel):
    subject: str
    filename: str
    title: str | None = None

class SummaryRequest(BaseModel):
    lesson: LessonRef

class KeyPointsRequest(BaseModel):
    lesson: LessonRef

class QuizGenRequest(BaseModel):
    lesson: LessonRef
    count: int = 8
    difficulty: str = "Medium"

@app.get("/")
def read_root():
    return {"status": "AI Service is running"}

@app.post("/ai/chat")
def chat(request: ChatRequest):
    if _IMPORT_ERROR is not None or tutor_answer is None:
        return {
            "response": (
                "AI service is running, but advanced tutor features are not available yet.\n\n"
                "Fix: install dependencies in `ai-service/requirements.txt` and restart the AI service."
            )
        }

    # Best-effort: if lesson ref provided, ground using the PDF context.
    lesson = request.lesson or {}
    ctx = None
    try:
        subject = str(lesson.get("subject") or "")
        filename = str(lesson.get("filename") or "")
        if subject and filename:
            pdf_path = ROOT / "pdf_library" / subject / filename
            if pdf_path.exists() and build_context is not None:
                ctx = build_context([pdf_path], max_chars=24_000)
    except Exception:
        ctx = None

    try:
        response = tutor_answer(request.message, pdf_context=ctx, mode="Use Current PDF" if ctx else "Universal Knowledge")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/summary")
def summary(req: SummaryRequest):
    if _IMPORT_ERROR is not None or generate_summary is None or build_context is None:
        raise HTTPException(
            status_code=503,
            detail="Summary endpoint unavailable. Install AI dependencies and restart the AI service.",
        )

    pdf_path = ROOT / "pdf_library" / req.lesson.subject / req.lesson.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Lesson PDF not found.")

    try:
        ctx = build_context([pdf_path], max_chars=24_000)
        out = generate_summary(ctx, language="English", depth=2)
        return {"summary": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/key-points")
def key_points(req: KeyPointsRequest):
    if _IMPORT_ERROR is not None or generate_key_points is None or build_context is None:
        raise HTTPException(
            status_code=503,
            detail="Key points endpoint unavailable. Install AI dependencies and restart the AI service.",
        )

    pdf_path = ROOT / "pdf_library" / req.lesson.subject / req.lesson.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Lesson PDF not found.")

    try:
        ctx = build_context([pdf_path], max_chars=24_000)
        out = generate_key_points(ctx)
        return {"keyPoints": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/quiz")
def quiz(req: QuizGenRequest):
    if _IMPORT_ERROR is not None or generate_pdf_quiz is None or QuizRequest is None or build_context is None:
        raise HTTPException(
            status_code=503,
            detail="Quiz endpoint unavailable. Install AI dependencies and restart the AI service.",
        )

    pdf_path = ROOT / "pdf_library" / req.lesson.subject / req.lesson.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Lesson PDF not found.")

    try:
        ctx = build_context([pdf_path], max_chars=24_000)
        qreq = QuizRequest(quiz_type="Mixed", count=max(3, min(25, req.count)), difficulty=req.difficulty)
        out = generate_pdf_quiz(ctx, qreq)
        return {"quiz": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Make sure to run exactly this port for the frontend to connect
    uvicorn.run(app, host="0.0.0.0", port=8000)
