from __future__ import annotations

import json
from dataclasses import dataclass

from .ai_tutor import answer_question, LlmConfig


# =========================================================
# QUIZ REQUEST
# =========================================================

@dataclass(frozen=True)
class QuizRequest:

    quiz_type: str = "MCQ"

    count: int = 10

    difficulty: str = "Medium"


# =========================================================
# QUIZ SYSTEM PROMPT
# =========================================================

QUIZ_SYSTEM_PROMPT = """
You are an elite educational quiz generator.

Your job:
- Create highly accurate educational quizzes.
- Use professional English only.
- Make quizzes interactive and student friendly.
- Questions should test understanding, not memorization only.
- Include real-world thinking where possible.

IMPORTANT:
Return ONLY valid JSON.

JSON FORMAT:

{
  "title": "Quiz title",
  "questions": [
    {
      "question": "Question text",
      "options": {
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      },
      "answer": "B",
      "explanation": "Explanation here"
    }
  ]
}

RULES:
- Do NOT return markdown.
- Do NOT return code blocks.
- Do NOT return explanations outside JSON.
- Every question MUST contain:
  - question
  - options
  - answer
  - explanation
"""


# =========================================================
# BUILD PROMPT
# =========================================================

def build_quiz_prompt(
    topic: str,
    req: QuizRequest,
    pdf_context: str | None = None,
) -> str:

    base = f"""
Generate a professional educational quiz.

Quiz Type:
{req.quiz_type}

Difficulty:
{req.difficulty}

Number of Questions:
{req.count}

Topic:
{topic}

Requirements:
- Use clear English.
- Avoid repeated questions.
- Questions should feel modern and interactive.
- Add concise explanations.
- Keep explanations educational.
"""

    if pdf_context and pdf_context.strip():

        base += f"""

Use this uploaded study material as the PRIMARY source:

{pdf_context[:12000]}
"""

    return base


# =========================================================
# CLEAN JSON RESPONSE
# =========================================================

def clean_json_response(text: str) -> str:

    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "")

    if text.startswith("```"):
        text = text.replace("```", "")

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# =========================================================
# GENERATE QUIZ
# =========================================================
def generate_quiz_data(
    topic: str,
    req: QuizRequest,
    pdf_context: str | None = None,
):

    prompt = f"""
{QUIZ_SYSTEM_PROMPT}

STRICT INSTRUCTIONS:

You MUST return ONLY valid JSON.

DO NOT:
- add markdown
- add intro text
- add explanations outside JSON
- add headings outside JSON
- add ```json

RETURN RAW JSON ONLY.

Topic:
{topic}

Quiz Type:
{req.quiz_type}

Difficulty:
{req.difficulty}

Number of Questions:
{req.count}
"""

    if pdf_context and pdf_context.strip():

        prompt += f"""

Use this study material:

{pdf_context[:12000]}
"""

    response = answer_question(
        question=prompt,
        pdf_context=None,
        config=LlmConfig(
            temperature=0.2,
            max_tokens=2500,
        ),
    )

    response = clean_json_response(response)

    # ============================================
    # FIX JSON EXTRA TEXT ISSUE
    # ============================================

    try:

        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1:
            response = response[start:end+1]

        data = json.loads(response)

        if not isinstance(data, dict):
            raise ValueError("Invalid JSON")

        if "questions" not in data:
            raise ValueError("Missing questions")

        return data

    except Exception as e:

        print("QUIZ JSON ERROR:")
        print(e)
        print(response)

        return {
            "title": "Quiz Generation Failed",
            "questions": [
                {
                    "question": "AI failed to generate valid quiz JSON.",
                    "options": {
                        "A": "Retry",
                        "B": "Refresh",
                        "C": "Check AI response",
                        "D": "All of the above"
                    },
                    "answer": "D",
                    "explanation": str(response)[:700]
                }
            ]
        }
# =========================================================
# UNIVERSAL QUIZ
# =========================================================

def generate_universal_quiz(
    topic: str,
    req: QuizRequest,
):

    topic = (topic or "").strip()

    if not topic:
        topic = "General Knowledge"

    return generate_quiz_data(
        topic=topic,
        req=req,
        pdf_context=None,
    )


# =========================================================
# PDF QUIZ
# =========================================================

def generate_pdf_quiz(
    pdf_text: str,
    req: QuizRequest,
):

    return generate_quiz_data(
        topic="Uploaded PDF Lesson",
        req=req,
        pdf_context=pdf_text,
    )


# =========================================================
# STREAMLIT QUIZ RENDERER
# =========================================================
def render_quiz_streamlit(st, quiz_data):

    st.markdown(
        f"# {quiz_data.get('title', 'Quiz')}"
    )

    questions = quiz_data.get("questions", [])

    if not questions:
        st.error("No questions generated.")
        return

    # =====================================================
    # SESSION STATE
    # =====================================================

    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0

    if "quiz_answered" not in st.session_state:
        st.session_state["quiz_answered"] = {}

    # =====================================================
    # QUESTIONS
    # =====================================================

    for i, q in enumerate(questions):

        st.markdown("---")

        question = q.get("question", "")

        options = q.get("options", {})

        answer = q.get("answer", "")

        explanation = q.get("explanation", "")

        st.markdown(
            f"## Question {i+1}"
        )

        st.write(question)

        option_values = [
            f"{k}. {v}"
            for k, v in options.items()
        ]

        selected = st.radio(
            "Choose your answer:",
            option_values,
            key=f"quiz_option_{i}",
        )

        already_answered = (
            i in st.session_state["quiz_answered"]
        )

        if not already_answered:

            if st.button(
                f"Submit Answer {i+1}",
                key=f"submit_{i}",
            ):

                selected_key = selected.split(".")[0]

                correct_text = (
                    f"{answer}. "
                    f"{options.get(answer, '')}"
                )

                is_correct = (
                    selected_key == answer
                )

                st.session_state["quiz_answered"][i] = {
                    "selected": selected_key,
                    "correct": answer,
                    "is_correct": is_correct,
                }

                if is_correct:

                    st.session_state["quiz_score"] += 1

                st.rerun()

        # =================================================
        # SHOW RESULTS
        # =================================================

        if already_answered:

            result = st.session_state["quiz_answered"][i]

            correct_text = (
                f"{answer}. "
                f"{options.get(answer, '')}"
            )

            if result["is_correct"]:

                st.success(
                    f"✅ Correct! {correct_text}"
                )

            else:

                st.error(
                    f"❌ Wrong Answer"
                )

                st.success(
                    f"✅ Correct Answer: {correct_text}"
                )

            st.info(
                f"📘 Explanation:\n\n{explanation}"
            )

    # =====================================================
    # FINAL SCORE
    # =====================================================

    st.markdown("---")

    st.metric(
        "Final Score",
        f"{st.session_state['quiz_score']}/{len(questions)}"
    )

    # =====================================================
    # RESET QUIZ
    # =====================================================
    if st.button("Reset Quiz"):

        st.session_state["quiz_score"] = 0

        st.session_state["quiz_answered"] = {}

        st.rerun()

    return {
        "correct": st.session_state["quiz_score"],
        "total": len(questions)
    }