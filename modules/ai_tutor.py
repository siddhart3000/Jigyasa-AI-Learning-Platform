from __future__ import annotations

import os
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class LlmConfig:

    # UPDATED MODEL (OLD ONE REMOVED BY GROQ)
    model: str = "llama-3.3-70b-versatile"

    temperature: float = 0.5
    max_tokens: int = 1400


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Jigyasa AI Tutor, an elite AI teacher designed to teach students clearly and intelligently.

CORE BEHAVIOR:
- Always respond in professional English.
- Teach like an experienced real-world teacher.
- Explain difficult topics simply.
- Use structured formatting.
- Use headings, bullets, and spacing.
- Use real-world examples.
- Use analogies and memory tricks.
- Keep explanations engaging and readable.
- Avoid robotic responses.
- Avoid unnecessary complexity.

TEACHING STYLE:
- Start with a simple explanation.
- Then explain deeper concepts.
- Add examples from daily life.
- Add tricks to remember concepts.
- End with a quick recap when useful.

PDF BEHAVIOR:
- If PDF context exists, prioritize it.
- If the answer is not directly available in the PDF:
say:
"This is not directly mentioned in the uploaded document, but here is the concept clearly explained."

OUTPUT STYLE:
- Clean markdown formatting
- Modern conversational teaching style
- Never use Hinglish or Hindi
- Never mention internal AI limitations unnecessarily
"""


# =========================================================
# GROQ CLIENT
# =========================================================

def _client():

    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY in .env file."
        )

    return Groq(api_key=api_key)


@lru_cache(maxsize=1)
def _cached_client():
    return _client()


# =========================================================
# API TEST
# =========================================================

def test_api() -> str:

    try:

        client = _cached_client()

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": "Say hello professionally."
                }
            ],
            temperature=0.3,
            max_tokens=50,
        )

        return (
            response.choices[0]
            .message.content
            .strip()
        )

    except Exception as e:

        return f"API ERROR: {e}"


# =========================================================
# MAIN AI RESPONSE
# =========================================================

def answer_question(
    question: str,
    pdf_context: Optional[str] = None,
    chat_history: Optional[list] = None,
    config: Optional[LlmConfig] = None,
) -> str:

    config = config or LlmConfig()

    question = question.strip()

    if not question:
        return "Please enter a question."

    context = (pdf_context or "").strip()

    # =====================================================
    # USER PROMPT
    # =====================================================

    if context:

        user_prompt = f"""
Use the uploaded study material below as the PRIMARY CONTEXT.

UPLOADED STUDY MATERIAL:
{context[:12000]}

STUDENT QUESTION:
{question}

Instructions:
- Answer like an elite teacher.
- Use examples.
- Explain concepts step-by-step.
- Make difficult ideas simple.
"""

    else:

        user_prompt = f"""
STUDENT QUESTION:
{question}

Instructions:
- Teach clearly.
- Use real-world examples.
- Add memory tricks when useful.
- Explain step-by-step.
"""

    # =====================================================
    # BUILD MESSAGE HISTORY
    # =====================================================

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    if chat_history:

        for msg in chat_history[-10:]:

            role = msg.get("role", "user")
            content = msg.get("content", "")

            if content.strip():

                messages.append({
                    "role": role,
                    "content": content
                })

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    # =====================================================
    # API CALL
    # =====================================================

    try:

        client = _cached_client()

        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        answer = (
            response.choices[0]
            .message.content
            .strip()
        )

        if not answer:
            return "No response generated."

        return answer

    except Exception as e:

        error_text = str(e)

        # CLEANER USER ERRORS
        if "rate_limit" in error_text.lower():

            return """
⚠️ AI service is currently busy.

Please wait a few seconds and try again.
"""

        if "invalid_api_key" in error_text.lower():

            return """
⚠️ Invalid GROQ API key.

Please check your .env configuration.
"""

        if "model_decommissioned" in error_text.lower():

            return """
⚠️ AI model is outdated.

Please restart the application with updated settings.
"""

        return f"""
⚠️ AI Tutor Error

{e}
"""


# =========================================================
# SUMMARY GENERATOR
# =========================================================

def generate_summary(
    pdf_context: str,
    config: Optional[LlmConfig] = None,
) -> str:

    text = pdf_context.strip()

    if not text:
        return "No PDF content available for summarization."

    config = config or LlmConfig(
        temperature=0.4,
        max_tokens=1200,
    )

    prompt = f"""
Summarize the following study material like an excellent teacher.

Requirements:
- Use simple English
- Explain clearly
- Keep it engaging
- Add important concepts
- Add examples if useful
- End with a quick recap

STUDY MATERIAL:
{text[:12000]}
"""

    return answer_question(
        question=prompt,
        pdf_context=None,
        config=config,
    )


# =========================================================
# KEY POINTS GENERATOR
# =========================================================

def generate_key_points(
    pdf_context: str,
    config: Optional[LlmConfig] = None,
) -> str:

    text = pdf_context.strip()

    if not text:
        return "No content available."

    config = config or LlmConfig(
        temperature=0.3,
        max_tokens=900,
    )

    prompt = f"""
Extract the most important key points from the study material.

Requirements:
- Return concise bullet points
- Include definitions
- Include formulas if present
- Include important concepts
- Keep points readable

STUDY MATERIAL:
{text[:12000]}
"""

    return answer_question(
        question=prompt,
        pdf_context=None,
        config=config,
    )


# =========================================================
# SINGLE TUTOR ENTRY POINT
# =========================================================

def tutor_answer(
    question: str,
    pdf_context: Optional[str] = None,
    chat_history: Optional[list] = None,
) -> str:

    return answer_question(
        question=question,
        pdf_context=pdf_context,
        chat_history=chat_history,
    )