from __future__ import annotations

from pathlib import Path

import streamlit as st
import os
import pandas as pd

from modules.ai_tutor import answer_question, generate_key_points, generate_summary, tutor_answer
from modules.analytics import (
    build_analytics_frame,
    detect_strength_band,
    engagement_score,
    learning_score,
    predict_exam_score,
    quiz_accuracy,
    study_consistency_score,
    topic_mastery,
)
from modules.leaderboard import leaderboard_df
from modules.pdf_reader import (
    build_context,
    create_class,
    create_subject,
    format_class_display,
    format_pdf_display,
    format_subject_display,
    list_classes,
    list_pdfs,
    list_subjects,
    save_uploaded_pdf,
)
from modules.auth import login_user, signup_user
from modules.quiz_generator import (
    QuizRequest,
    generate_pdf_quiz,
    generate_universal_quiz,
    render_quiz_streamlit,
    generate_quiz_data,
)
from modules.storage import log_daily_activity, save_student, upsert_student_profile
from modules.ui_components import sidebar_nav
from modules.videos import learning_videos
from modules.theme import apply_theme
from modules.rag_engine import add_document_to_index, search_context



st.set_page_config(
    page_title="Jigyasa: AI Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

def production_setup():
    """Ensures directories and NLTK data exist for Streamlit Cloud."""
    # 1. Create necessary directories
    for folder in ["uploads", "AI_Tutor", "pdf_library"]:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    # 2. Setup NLTK
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.1rem; padding-bottom: 3rem; }
          [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
          .ald-card { border:1px solid rgba(255,255,255,0.10); border-radius:16px; padding:16px;
                      background: rgba(255,255,255,0.03); }
          .ald-card h4 { margin:0; padding:0; font-size: 16px; font-weight: 700; }
          .ald-muted { opacity: 0.75; font-size: 13px; }
          .ald-row { display:flex; gap:12px; overflow-x:auto; padding-bottom: 6px; }
          .ald-tile { min-width: 220px; max-width: 260px; }
          .ald-hero-title { font-size: 2.1rem; font-weight: 800; letter-spacing: 0.03em;
                            background: linear-gradient(90deg,#8be9fd,#ff79c6);
                            -webkit-background-clip: text; color: transparent; }
          .ald-hero-sub { font-size: 0.98rem; opacity:0.8; }
          .ald-ring {
             width: 140px; height: 140px; border-radius: 50%;
             background:
               conic-gradient(#50fa7b calc(var(--val)*1%), rgba(80,250,123,0.12) 0);
             display:flex; align-items:center; justify-content:center;
             box-shadow: 0 0 18px rgba(80,250,123,0.45);
          }
          .ald-ring-inner {
             width: 102px; height: 102px; border-radius:50%;
             background: rgba(10,10,10,0.85);
             display:flex; align-items:center; justify-content:center;
             font-size: 1.5rem; font-weight:700;
          }

          /* Workspace layout helpers */
          .ald-panel {
            background: color-mix(in srgb, var(--card) 92%, transparent) !important;
            border: 1px solid color-mix(in srgb, var(--text) 12%, transparent) !important;
            border-radius: 16px !important;
            padding: 14px !important;
            box-shadow: 0 14px 45px rgba(0,0,0,0.22);
          }
          .ald-sticky {
            position: sticky;
            top: 0.85rem;
            align-self: start;
          }
          .ald-chapter-btn .stButton>button {
            width: 100% !important;
            text-align: left !important;
            justify-content: flex-start !important;
            border-radius: 14px !important;
            padding: 10px 12px !important;
            border: 1px solid color-mix(in srgb, var(--text) 12%, transparent) !important;
            background: color-mix(in srgb, var(--card) 88%, transparent) !important;
            color: var(--text) !important;
          }
          .ald-chapter-btn .stButton>button:hover {
            border-color: color-mix(in srgb, var(--accent) 35%, transparent) !important;
            box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 14%, transparent) !important;
          }
          .ald-active .stButton>button {
            border-color: color-mix(in srgb, var(--accent) 55%, transparent) !important;
            background: color-mix(in srgb, var(--accent) 12%, var(--card) 88%) !important;
          }
          .sidebar-user-card {
            margin-top: 2rem;
            border: 1px solid rgba(139, 233, 253, 0.2);
            border-radius: 12px;
            padding: 12px;
            background: rgba(255,255,255,0.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3), 0 0 8px rgba(139, 233, 253, 0.1);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pdf_viewer(path: Path) -> None:
    try:
        from streamlit_pdf_viewer import pdf_viewer  # type: ignore

        pdf_viewer(str(path), height=700)
    except Exception:
        st.info("PDF viewer not available. Install `streamlit-pdf-viewer` to render PDFs inside the app.")
        st.download_button("Download PDF", data=path.read_bytes(), file_name=path.name, mime="application/pdf")


# Helper function for AI calls with error handling
def _call_ai_service(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        st.warning("⚠️ AI service is busy. Please wait a few seconds and try again.")
        return None

@st.cache_data(show_spinner=False)
def _cached_context(path_str: str, max_chars: int = 24_000) -> str:
    return build_context([Path(path_str)], max_chars=max_chars)


def get_active_student():
    email = st.session_state.get("student_email")
    name = st.session_state.get("student_name")
    return upsert_student_profile(email=email, name=name)

def page_login():
    st.markdown("<br>"*2, unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        st.markdown("<div class='ald-hero-title' style='text-align:center;'>Welcome Back</div>", unsafe_allow_html=True)
        with st.container(border=True):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login", type="primary", width="stretch"):
                success, res = login_user(email, password)
                if success:
                    st.session_state["logged_in"] = True
                    st.session_state["student_email"] = res.email
                    st.session_state["student_name"] = res.name
                    st.success("Redirecting...")
                    st.rerun()
                else:
                    st.error(res)
            if st.button("Need an account? Sign up", key="go_signup", width="stretch"):
                st.session_state["auth_page"] = "signup"
                st.rerun()

def page_signup():
    st.markdown("<br>"*2, unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        st.markdown("<div class='ald-hero-title' style='text-align:center;'>Create Account</div>", unsafe_allow_html=True)
        with st.container(border=True):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            p1 = st.text_input("Password", type="password")
            p2 = st.text_input("Confirm Password", type="password")
            if st.button("Sign Up", type="primary", width="stretch"):
                if p1 != p2:
                    st.error("Passwords do not match.")
                elif len(p1) < 6:
                    st.error("Password too short.")
                elif not name or not email:
                    st.error("Please fill all fields.")
                else:
                    success, msg = signup_user(name, email, p1)
                    if success:
                        st.session_state["logged_in"] = True
                        st.session_state["student_email"] = email.lower().strip()
                        st.session_state["student_name"] = name
                        st.success("Account created!")
                        st.rerun()
                    else:
                        st.error(msg)
            if st.button("Already have an account? Login", key="go_login", width="stretch"):
                st.session_state["auth_page"] = "login"
                st.rerun()


def sidebar_profile() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom: 5px; margin-top: -10px;">
                <h2 style="margin: 0; color: #8be9fd; font-size: 1.6rem;">🎓 Jigyasa</h2>
                <div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 12px;">AI Learning Platform</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def _mark_lesson(student) -> None:
    student.lessons_completed += 1
    student.xp += 15
    save_student(student)
    st.toast("Lesson completed! +15 XP")


def page_home(student) -> None:
    st.markdown("## ")
    col_hero, col_ring = st.columns([0.65, 0.35], vertical_alignment="center")
    with col_hero:
        st.markdown(
            "<div class='ald-hero-title'>Jigyasa – AI Learning Platform</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='ald-hero-sub'>Personalized AI powered learning system for notes, videos, quizzes, and your own AI tutor.</div>",
            unsafe_allow_html=True,
        )

    lessons = student.lessons_completed
    videos = student.videos_watched
    quizzes = student.quizzes_attempted
    ai_qs = student.ai_questions_asked
    completion = min(100, int((lessons * 8 + videos * 4 + quizzes * 10 + ai_qs * 1) / 5))

    with col_ring:
        score = learning_score(student)
        st.markdown(
            f"<div class='ald-ring' style='--val:{min(100, max(0,int(score)))};'><div class='ald-ring-inner'>{int(score)}</div></div>",
            unsafe_allow_html=True,
        )
        st.caption("Learning Score")

    st.markdown("### Today’s snapshot")
    a, b, c, d, e = st.columns(5)
    a.metric("Learning Score", f"{learning_score(student):.1f}")
    b.metric("Study time (min)", student.study_time_minutes)
    c.metric("Videos watched", videos)
    d.metric("Quiz accuracy", f"{quiz_accuracy(student):.1f}%")
    e.metric("AI Tutor usage", student.ai_questions_asked)

    with st.container(border=True):
        st.markdown("### Weekly activity")
        df = build_analytics_frame(student)
        chart_df = df[df["Metric"].isin(["Quiz Accuracy (%)", "AI Questions", "Videos Watched", "Practice Attempts"])].copy()
        chart_df = chart_df.set_index("Metric")
        left, right = st.columns([0.7, 0.3], vertical_alignment="center")
        with left:
            st.bar_chart(chart_df)
        with right:
            st.markdown("**Overall progress**")
            st.progress(completion / 100)
            st.markdown(f"Learning Score: **{learning_score(student):.2f}**")
            st.caption(detect_strength_band(learning_score(student)))

    st.markdown("### Continue learning")
    classes = list_classes()
    default_class = classes[0] if classes else "Class_1"
    subjects = list_subjects(default_class) or ["English", "Maths", "Science"]
    cols = st.columns(min(4, len(subjects)))
    for i, subj in enumerate(subjects[:4]):
        pdfs = list_pdfs(default_class, subj)
        total = max(1, len(pdfs))
        prog = min(1.0, lessons / (total * 2))
        with cols[i]:
            st.markdown(f"<div class='ald-card'><h4>📘 {subj}</h4><div class='ald-muted'>{len(pdfs)} chapters</div></div>", unsafe_allow_html=True)
            st.progress(prog)
            st.caption(f"{int(prog*100)}% complete")

    st.divider()
    st.markdown("### Quick navigation")
    nav_cards = [
        ("📚", "Learning", "Open Notes & Videos", "learning"),
        ("🤖", "AI Tutor", "Ask anything, anytime", "tutor"),
        ("📝", "Quiz", "Practice and improve", "quiz"),
        ("📊", "Analytics", "See your insights", "analytics"),
        ("🏆", "Leaderboard", "Compete and grow", "leaderboard"),
    ]
    cols = st.columns(5)
    for idx, (ico, title, sub, key) in enumerate(nav_cards):
        with cols[idx]:
            with st.container(border=True):
                st.markdown(f"**{ico} {title}**")
                st.caption(sub)
                if st.button(f"Open {title}", key=f"home_nav_{key}", width="stretch"):
                    # sidebar uses nav_label; set it so navigation switches cleanly
                    label_map = {
                        "learning": "📚  Learning",
                        "tutor": "🤖  AI Tutor",
                        "quiz": "📝  Quiz",
                        "analytics": "📊  Analytics",
                        "leaderboard": "🏆  Leaderboard",
                        "profile": "👤  Profile",
                        "home": "🏠  Home",
                    }
                    st.session_state["nav_label"] = label_map[key]
                    st.rerun()


def page_learning(student) -> None:
    st.markdown("## Learning")
    tabs = st.tabs(["🗒️ Notes", "🎬 Videos"])
    with tabs[0]:
        page_notes(student)
    with tabs[1]:
        page_videos(student)


def page_library(student) -> None:
    # Legacy; routed via Learning → Notes now.
    page_notes(student)

    classes = list_classes()
    if not classes:
        st.info("No classes found yet. Upload PDFs in Notes.")
        return
    subjects = list_subjects(classes[0])
    if not subjects:
        st.info("No subjects found yet. Upload PDFs in Notes.")
        return

    tabs = st.tabs([f"📚 {s}" for s in subjects])
    for idx, subject in enumerate(subjects):
        with tabs[idx]:
            pdfs = list_pdfs(classes[0], subject)
            if not pdfs:
                st.warning("No chapters yet.")
                continue

            # Chapter cards
            per_row = 4
            for start in range(0, len(pdfs), per_row):
                row = st.columns(per_row)
                for j, doc in enumerate(pdfs[start : start + per_row]):
                    chapter_name = f"Chapter {start + j + 1}"
                    with row[j]:
                        with st.container(border=True):
                            st.markdown(f"**{chapter_name}**")
                            st.caption(doc.path.name)
                            if st.button("Open", key=f"lib_open_{subject}_{doc.path.name}", width="stretch"):
                                st.session_state["lib_active_path"] = str(doc.path)

            active = st.session_state.get("lib_active_path")
            if active:
                path = Path(active)
                st.divider()
                top = st.columns([0.7, 0.3], vertical_alignment="center")
                top[0].markdown(f"### 📄 {path.name}")
                top[1].button("Mark completed (+15 XP)", width="stretch", on_click=_mark_lesson, args=(student,))
                _pdf_viewer(path)


def page_ai_tutor(student) -> None:

    st.markdown("""
    <style>

    .chat-header {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 5px;
        background: linear-gradient(90deg,#8be9fd,#bd93f9,#ff79c6);
        -webkit-background-clip: text;
        color: transparent;
        letter-spacing: -1px;
    }

    .chat-sub {
        opacity: 0.75;
        margin-bottom: 25px;
        font-size: 1rem;
        line-height: 1.6;
    }

    .upload-box {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 18px;
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        margin-bottom: 24px;
    }

    .feature-card {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 18px;
        background: rgba(255,255,255,0.025);
        transition: 0.2s ease;
        min-height: 130px;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        border: 1px solid rgba(139,233,253,0.25);
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }

    .welcome-card {
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 24px;
        padding: 22px;
        background: linear-gradient(
            135deg,
            rgba(139,233,253,0.08),
            rgba(189,147,249,0.05)
        );
        margin-bottom: 25px;
    }

    .chat-tip {
        opacity: 0.7;
        font-size: 0.92rem;
        margin-top: 6px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='chat-header'>AI Tutor</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class='chat-sub'>
    Your personal AI teacher for deep learning, concept mastery,
    PDF learning, exam preparation, real-world examples,
    memory tricks, and intelligent explanations.
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SESSION STATES
    # ---------------------------------------------------------

    if "tutor_messages" not in st.session_state:
        st.session_state["tutor_messages"] = []

    if "active_pdf_context" not in st.session_state:
        st.session_state["active_pdf_context"] = None

    if "last_uploaded_pdf" not in st.session_state:
        st.session_state["last_uploaded_pdf"] = None

    # ---------------------------------------------------------
    # CACHED PDF CONTEXT
    # ---------------------------------------------------------

    @st.cache_data(show_spinner=False)
    def cached_pdf_context(path_str: str):
        return build_context([Path(path_str)])

    # ---------------------------------------------------------
    # PDF UPLOAD SECTION
    # ---------------------------------------------------------

    with st.container():

        st.markdown(
            "<div class='upload-box'>",
            unsafe_allow_html=True
        )

        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="ai_tutor_pdf",
            label_visibility="collapsed"
        )

        st.caption(
            "📄 Upload notes, books, research papers, assignments, or study material."
        )

        if uploaded_pdf:

            if st.session_state["last_uploaded_pdf"] != uploaded_pdf.name:

                pdf_path = save_uploaded_pdf(
                    "AI_Tutor",
                    uploaded_pdf.name,
                    uploaded_pdf.getvalue()
                )

                with st.spinner("Reading and understanding PDF..."):

                    pdf_context = cached_pdf_context(str(pdf_path))

                st.session_state["active_pdf_context"] = pdf_context
                st.session_state["last_uploaded_pdf"] = uploaded_pdf.name

                st.success(f"PDF loaded successfully: {uploaded_pdf.name}")

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # ---------------------------------------------------------
    # FEATURE CARDS
    # ---------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='feature-card'>
        <h4>🧠 Smart Teaching</h4>
        <p>
        Learn concepts step-by-step with deep explanations
        and easy understanding.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='feature-card'>
        <h4>⚡ Memory Tricks</h4>
        <p>
        Get shortcuts, mnemonics, and smart tricks
        to remember concepts faster.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='feature-card'>
        <h4>📚 PDF Learning</h4>
        <p>
        Upload any PDF and ask unlimited questions
        from your study material.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # WELCOME STATE
    # ---------------------------------------------------------

    if len(st.session_state["tutor_messages"]) == 0:

        st.markdown("""
        <div class='welcome-card'>
            <h3>🚀 Start Learning</h3>
            <p>
            Ask anything. Learn with examples, tricks,
            visual understanding, and real teacher-style explanations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button(
                "Explain Quantum Physics",
                width="stretch"
            ):
                st.session_state["example_prompt"] = (
                    "Explain Quantum Physics simply"
                )

        with c2:
            if st.button(
                "Teach with Real Examples",
                width="stretch"
            ):
                st.session_state["example_prompt"] = (
                    "Teach Artificial Intelligence with real-world examples"
                )

        with c3:
            if st.button(
                "Summarize my PDF",
                width="stretch"
            ):
                st.session_state["example_prompt"] = (
                    "Summarize this PDF in simple language"
                )

    # ---------------------------------------------------------
    # RENDER CHAT
    # ---------------------------------------------------------

    for msg in st.session_state["tutor_messages"][-20:]:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

    # ---------------------------------------------------------
    # CHAT INPUT
    # ---------------------------------------------------------

    default_prompt = st.session_state.get(
        "example_prompt",
        ""
    )

    prompt = st.chat_input(
        "Ask anything about any topic or uploaded PDF..."
    )

    if not prompt and default_prompt:
        prompt = default_prompt
        st.session_state["example_prompt"] = ""

    # ---------------------------------------------------------
    # PROCESS MESSAGE
    # ---------------------------------------------------------

    if prompt:

        st.session_state["tutor_messages"].append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):

            st.markdown(prompt)

        # -----------------------------------------------------
        # UPDATE STUDENT STATS
        # -----------------------------------------------------

        student.ai_questions_asked += 1
        student.xp += 2

        save_student(student)

        # -----------------------------------------------------
        # PDF CONTEXT
        # -----------------------------------------------------

        pdf_context = st.session_state.get(
            "active_pdf_context",
            None
        )

        # -----------------------------------------------------
        # ADVANCED TEACHING PROMPT
        # -----------------------------------------------------

        teaching_prompt = f"""
You are Jigyasa AI.

You are a world-class AI teacher.

STRICT RULES:

- Always answer ONLY in English.
- Never use Hindi or Hinglish.
- Teach like an elite private tutor.
- Use very simple explanations.
- Explain step-by-step.
- Use bullet points.
- Use headings.
- Add real-world examples.
- Add memory tricks.
- Add practical applications.
- Make learning engaging.
- Make difficult concepts easy.
- Never give robotic answers.
- Never answer too shortly.
- If PDF exists, prioritize PDF knowledge.
- If answer is outside PDF,
  use your own knowledge intelligently.

Student Question:
{prompt}
"""

        # -----------------------------------------------------
        # AI RESPONSE
        # -----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("Thinking deeply..."):

                answer = _call_ai_service(
                    answer_question,
                    teaching_prompt,
                    pdf_context=pdf_context
                )

                if not answer:

                    answer = """
⚠️ AI service is temporarily unavailable.

Please try again in a few seconds.
"""

            st.markdown(answer)

        # -----------------------------------------------------
        # SAVE CHAT
        # -----------------------------------------------------

        st.session_state["tutor_messages"].append({
            "role": "assistant",
            "content": answer
        })
def page_quiz(student) -> None:

    from modules.quiz_generator import (
        QuizRequest,
        generate_universal_quiz,
        generate_pdf_quiz,
        render_quiz_streamlit,
    )

    st.markdown("""
    <style>

    .quiz-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 8px;
        background: linear-gradient(90deg,#8be9fd,#bd93f9,#ff79c6);
        -webkit-background-clip: text;
        color: transparent;
    }

    .quiz-sub {
        opacity: 0.75;
        margin-bottom: 25px;
        font-size: 1rem;
    }

    .quiz-box {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 22px;
        background: rgba(255,255,255,0.03);
        margin-bottom: 20px;
    }

    .quiz-info-card {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 16px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 16px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='quiz-title'>Interactive Quiz</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='quiz-sub'>Generate smart quizzes with instant evaluation, explanations, and learning feedback.</div>",
        unsafe_allow_html=True
    )

    # =========================================================
    # SESSION STATE
    # =========================================================

    if "quiz_data" not in st.session_state:
        st.session_state["quiz_data"] = None

    # =========================================================
    # PDF CONTEXT
    # =========================================================

    pdf_context = st.session_state.get(
        "active_pdf_context",
        None
    )

    # =========================================================
    # QUIZ SETTINGS
    # =========================================================

    with st.container():

        st.markdown(
            "<div class='quiz-box'>",
            unsafe_allow_html=True
        )

        mode = st.radio(
            "Quiz Source",
            [
                "Universal Topic",
                "Uploaded PDF"
            ],
            horizontal=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            quiz_type = st.selectbox(
                "Quiz Type",
                [
                    "MCQ",
                    "Conceptual",
                    "Application Based",
                    "Exam Style",
                ]
            )

        with col2:

            difficulty = st.selectbox(
                "Difficulty",
                [
                    "Easy",
                    "Medium",
                    "Hard"
                ],
                index=1
            )

        with col3:

            count = st.slider(
                "Questions",
                5,
                20,
                10
            )

        topic = st.text_input(
            "Topic",
            placeholder="e.g. Photosynthesis, AI, Freedom Fighters"
        )

        st.markdown(
            """
            <div class='quiz-info-card'>
            🧠 AI generates interactive questions with:
            <ul>
                <li>Instant evaluation</li>
                <li>Correct/incorrect highlighting</li>
                <li>Detailed explanations</li>
                <li>Learning reinforcement</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        generate = st.button(
            "Generate Interactive Quiz",
            type="primary",
            width="stretch"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # =========================================================
    # GENERATE QUIZ
    # =========================================================

    if generate:

        req = QuizRequest(
            quiz_type=quiz_type,
            count=count,
            difficulty=difficulty,
        )

        with st.spinner("Generating intelligent quiz..."):

            try:

                if mode == "Uploaded PDF":

                    if not pdf_context:

                        st.error(
                            "Please upload a PDF in AI Tutor first."
                        )

                        return

                    quiz_data = generate_pdf_quiz(
                        pdf_context,
                        req,
                    )

                else:

                    quiz_data = generate_universal_quiz(
                        topic,
                        req,
                    )

                st.session_state["quiz_data"] = quiz_data
                st.session_state["quiz_score_saved"] = False

                # ============================================
                # STUDENT STATS
                # ============================================

                student.quizzes_attempted += 1
                student.practice_attempts += 1
                student.xp += 10

                log_daily_activity(
                    student,
                    "quiz_attempts",
                    1
                )

                save_student(student)

            except Exception as e:

                st.error(
                    f"Quiz generation failed:\n\n{e}"
                )

       # =========================================================
    # RENDER INTERACTIVE QUIZ
    # =========================================================

    if st.session_state["quiz_data"]:

        st.markdown("## Generated Quiz")

        quiz_result = render_quiz_streamlit(
            st,
            st.session_state["quiz_data"]
        )

        if quiz_result:

            correct = quiz_result.get("correct", 0)
            total = quiz_result.get("total", 0)

            already_saved = st.session_state.get(
                "quiz_score_saved",
                False
            )

            if not already_saved:

                student.quiz_correct += correct
                student.quiz_total += total

                save_student(student)

                st.session_state["quiz_score_saved"] = True
def page_videos(student) -> None:
    st.markdown("## Learning Videos")
    st.caption("YouTube-style playlists: pick a subject and watch videos as cards.")

    vids = learning_videos()
    tabs = st.tabs([f"🎬 {k}" for k in vids.keys()])

    def yt_id(url: str) -> str | None:
        if "v=" in url:
            return url.split("v=", 1)[1].split("&", 1)[0]
        if "youtu.be/" in url:
            return url.split("youtu.be/", 1)[1].split("?", 1)[0]
        return None

    for idx, subject in enumerate(vids.keys()):
        with tabs[idx]:
            items = vids[subject]
            per_row = 3
            for start in range(0, len(items), per_row):
                row = st.columns(per_row)
                for j, v in enumerate(items[start : start + per_row]):
                    with row[j]:
                        watched = v["id"] in (student.watched_video_ids or [])
                        thumb_id = yt_id(v["url"])
                        thumb = f"https://img.youtube.com/vi/{thumb_id}/hqdefault.jpg" if thumb_id else None
                        with st.container(border=True):
                            if thumb:
                                st.image(thumb, use_container_width=True)
                            st.markdown(f"**{v['title']}**")
                            b1, b2 = st.columns([0.55, 0.45])
                            b1.link_button("Watch", v["url"], width="stretch")
                            if b2.button("Watched ✓" if watched else "+5 XP", key=f"vid_{v['id']}", width="stretch", disabled=watched):
                                student.videos_watched += 1
                                student.video_watch_count += 1
                                student.xp += 5
                                student.study_time_minutes += 5
                                student.video_watch_time_minutes += 5
                                log_daily_activity(student, "videos_watched", 1)
                                log_daily_activity(student, "video_minutes", 5)
                                student.watched_video_ids.append(v["id"])
                                save_student(student)
                                st.rerun()


# Placeholder for extractive summary (if summarize_pdf doesn't exist)
def summarize_pdf_extractive(context: str) -> str:
    # This is a simplified simulation. In a real scenario, this would use a research-based extractive summarization model.
    # For now, we'll use the LLM to extract important sentences.
    prompt = "Extract the most important sentences from the following text to form a concise summary. Do not add any new information or explanations. Only provide sentences directly from the text.\n\nText:\n" + context
    return _call_ai_service(answer_question, prompt, pdf_context=context)

# Placeholder for parsing quiz output (highly simplified)
    # This is a very basic parser and assumes a specific markdown format.
    # A robust solution would require more advanced NLP or a structured output from the LLM.
    # Example expected format:
    # ### Question X: [Question Text]
    # - [Option A]
    # 

def page_notes(student) -> None:
    with st.expander("Upload Notes"):
        up1, up2 = st.columns([0.35, 0.65])
        with up1:
            upload_subjects = list_subjects() or ["English", "Maths", "Science"]
            upload_subject = st.selectbox(
                "Subject",
                upload_subjects,
                format_func=format_subject_display,
                key="notes_upload_subject",
            )
            uploads = st.file_uploader("Upload PDF notes", type=["pdf"], accept_multiple_files=True, key="notes_uploads")
        if st.button("Save to library", type="primary", disabled=not uploads, width="stretch"):
            saved = []
            for f in uploads or []:
                p = save_uploaded_pdf(upload_subject, f.name, f.getvalue())
                saved.append(p.name)
                try:
                    with st.spinner(f"Indexing {p.name} into Knowledge Base..."): # type: ignore
                        add_document_to_index(p, subject=upload_subject, chapter="")
                except Exception:
                    pass
            st.success(f"Saved: {', '.join(saved)}")

    left, center, right = st.columns([1, 5, 1.2], vertical_alignment="top")

    subjects = list_subjects()
    if not subjects:
        create_subject("English")
        subjects = list_subjects()
    selected_subject = st.session_state.get("notes_subject_pick", subjects[0])
    if selected_subject not in subjects:
        selected_subject = subjects[0]

    with left:
        selected_subject = st.selectbox(
            "Subject",
            subjects,
            index=subjects.index(selected_subject) if selected_subject in subjects else 0,
            format_func=format_subject_display,
            key="notes_subject_pick",
        )

        pdfs = list_pdfs(selected_subject)
        current_active = st.session_state.get("notes_active_path", "")
        chapter_labels = [format_pdf_display(doc.path.name) for doc in pdfs]
        chapter_paths = [str(doc.path) for doc in pdfs]
        if chapter_labels:
            if current_active not in chapter_paths:
                current_active = chapter_paths[0]
            selected_chapter = st.radio(
                "Chapters",
                chapter_paths,
                index=chapter_paths.index(current_active),
                format_func=lambda p: chapter_labels[chapter_paths.index(p)],
                key="notes_chapter_pick",
            )
            if selected_chapter != st.session_state.get("notes_active_path", ""):
                # Clear all feature outputs when chapter changes
                st.session_state["notes_active_path"] = selected_chapter
                st.session_state["active_feature"] = None
                st.session_state["notes_summary_output_hybrid"] = ""
                st.session_state["notes_summary_output"] = ""
                st.session_state["notes_points_output"] = ""
                st.session_state["notes_lines_output"] = ""
                st.session_state["notes_mini_tutor_ans"] = ""
                st.session_state["notes_mini_q"] = ""
                st.session_state["notes_quiz_output"] = ""
                st.session_state["notes_quiz_answers"] = {}
                st.session_state["notes_quiz_submitted"] = False
                st.rerun()
        else:
            st.info("No chapters found.")

    with center:
        active_path = st.session_state.get("notes_active_path")
        if active_path and Path(active_path).exists():
            _pdf_viewer(Path(active_path))
        else:
            st.info("Select a chapter.")

    with right:
        if "active_feature" not in st.session_state:
            st.session_state["active_feature"] = None
        if "notes_summary_output_hybrid" not in st.session_state:
            st.session_state["notes_summary_output_hybrid"] = ""
        if "notes_summary_output_ai" not in st.session_state:
            st.session_state["notes_summary_output_ai"] = ""
        if "notes_points_output" not in st.session_state:
            st.session_state["notes_points_output"] = ""
        if "notes_lines_output" not in st.session_state:
            st.session_state["notes_lines_output"] = ""
        if "notes_quiz_raw_output" not in st.session_state:
            st.session_state["notes_quiz_raw_output"] = ""
        if "notes_quiz_score_saved" not in st.session_state:
            st.session_state["notes_quiz_score_saved"] = False
        if "notes_mini_tutor_ans" not in st.session_state:
            st.session_state["notes_mini_tutor_ans"] = ""

        active_path = st.session_state.get("notes_active_path")
        has_doc = bool(active_path and Path(active_path).exists())

        st.caption("Select an action:")
        col_btns = st.columns(2)

        def set_active_feature(feature_name):
            st.session_state["active_feature"] = feature_name
            st.session_state["notes_summary_output_hybrid"] = ""
            st.session_state["notes_summary_output_ai"] = ""
            st.session_state["notes_points_output"] = ""
            st.session_state["notes_lines_output"] = ""
            st.session_state["notes_quiz_raw_output"] = ""
            st.session_state["notes_quiz_score_saved"] = False
            st.session_state["notes_mini_tutor_ans"] = ""

        if col_btns[0].button("Summary", width="stretch", disabled=not has_doc, key="btn_summary"):
            set_active_feature("summary")
        if col_btns[1].button("Key Notes", width="stretch", disabled=not has_doc, key="btn_key_points"):
            set_active_feature("key_points")
        if col_btns[0].button("Imp. Lines", width="stretch", disabled=not has_doc, key="btn_important_lines"):
            set_active_feature("important_lines")
        if col_btns[1].button("Quiz", width="stretch", disabled=not has_doc, key="btn_quiz"):
            set_active_feature("quiz")
        if col_btns[0].button("Tutor", width="stretch", disabled=not has_doc, key="btn_tutor"):
            set_active_feature("tutor")

        st.divider()

        with st.container(height=500, border=True):
            if st.session_state["active_feature"] is None:
                st.markdown("Select an action to begin learning.")

            elif st.session_state["active_feature"] == "summary":
                st.markdown("### Summary")
                summary_mode = st.radio(
                    "Summary Mode",
                    ["⚡ Fast Hybrid", "🤖 AI Teacher"],
                    index=0,
                    key="notes_summary_mode_selector",
                    label_visibility="collapsed"
                )
                if st.button("Generate Summary", type="primary", width="stretch", key="gen_summary_btn"):
                    with st.spinner("Generating summary..."):
                        ctx = _cached_context(active_path, max_chars=24_000)
                        if summary_mode == "⚡ Fast Hybrid":
                            st.session_state["notes_summary_output_hybrid"] = summarize_pdf_extractive(ctx)
                            st.session_state["notes_summary_output_ai"] = ""
                        else:
                            st.session_state["notes_summary_output_ai"] = _call_ai_service(generate_summary, ctx, language="English", depth=3)
                            st.session_state["notes_summary_output_hybrid"] = ""

                if st.session_state["notes_summary_output_hybrid"]:
                    st.markdown("#### ⚡ Hybrid Summary")
                    st.markdown(st.session_state["notes_summary_output_hybrid"])
                elif st.session_state["notes_summary_output_ai"]:
                    st.markdown("#### 🤖 AI Teacher")
                    st.markdown(st.session_state["notes_summary_output_ai"])

            elif st.session_state["active_feature"] == "key_points":
                st.markdown("### Key Notes")
                if st.button("Generate Key Points", type="primary", width="stretch", key="gen_key_points_btn"):
                    with st.spinner("Extracting key points..."):
                        ctx = _cached_context(active_path, max_chars=24_000)
                        key_points_prompt = "Generate concise educational key points in English only. Use bullet points. No Hindi. No Hinglish. Keep explanations simple and educational.\n\nText:\n" + ctx
                        st.session_state["notes_points_output"] = _call_ai_service(answer_question, key_points_prompt, pdf_context=ctx)

                if st.session_state["notes_points_output"]:
                    st.markdown(st.session_state["notes_points_output"])

            elif st.session_state["active_feature"] == "important_lines":
                st.markdown("### Important Lines")
                if st.button("Find Important Lines", type="primary", width="stretch", key="gen_important_lines_btn"):
                    with st.spinner("Finding important lines..."):
                        ctx = _cached_context(active_path, max_chars=24_000)
                        line_prompt = "Extract the top 5-10 most important sentences from this chapter exactly as they appear in English only. Bullet points only."
                        st.session_state["notes_lines_output"] = _call_ai_service(answer_question, line_prompt, pdf_context=ctx)

                if st.session_state["notes_lines_output"]:
                    st.markdown(st.session_state["notes_lines_output"])

            elif st.session_state["active_feature"] == "quiz":
                st.markdown("### Lesson Quiz")
                if st.button("Generate Quiz", type="primary", width="stretch", key="gen_quiz_btn"):
                    with st.spinner("Generating quiz..."):
                        ctx = _cached_context(active_path, max_chars=24_000)
                        quiz_req = QuizRequest(quiz_type="MCQ", count=5, difficulty="Medium")
                        raw_quiz = _call_ai_service(generate_quiz_data, topic="Generate interactive MCQ quiz", req=quiz_req, pdf_context=ctx)
                        if raw_quiz:
                            st.session_state["notes_quiz_raw_output"] = raw_quiz
                            st.session_state["notes_quiz_score_saved"] = False
                            student.xp += 10
                            save_student(student)
                        else:
                            st.error("Quiz generation failed.")

                if st.session_state["notes_quiz_raw_output"]:
                    quiz_result = render_quiz_streamlit(st, st.session_state["notes_quiz_raw_output"])
                    if quiz_result:
                        correct = quiz_result.get("correct", 0)
                        total = quiz_result.get("total", 0)
                        if not st.session_state.get("notes_quiz_score_saved"):
                            student.quiz_correct += correct
                            student.quiz_total += total
                            student.quizzes_attempted += 1
                            student.practice_attempts += 1
                            log_daily_activity(student, "quiz_attempts", 1)
                            log_daily_activity(student, "quiz_questions", total)
                            save_student(student)
                            st.session_state["notes_quiz_score_saved"] = True
                            st.success(f"Score Saved: {correct}/{total}")

            elif st.session_state["active_feature"] == "tutor":
                st.markdown("### AI Mini-Tutor")
                mini_q = st.text_input("Ask about this chapter:", key="notes_mini_q_input")
                if st.button("Ask AI", width="stretch", disabled=not (has_doc and mini_q), key="ask_mini_tutor_btn"):
                    with st.spinner("Thinking..."):
                        ctx = _cached_context(active_path, max_chars=24_000)
                        tutor_prompt = f"Answer based on context in English only. Question: {mini_q}"
                        st.session_state["notes_mini_tutor_ans"] = _call_ai_service(answer_question, tutor_prompt, pdf_context=ctx)
                        if st.session_state["notes_mini_tutor_ans"]:
                            student.ai_questions_asked += 1
                            student.xp += 2
                            log_daily_activity(student, "ai_questions", 1)
                            save_student(student)

                if st.session_state["notes_mini_tutor_ans"]:
                    with st.chat_message("assistant"):
                        st.markdown(st.session_state["notes_mini_tutor_ans"])


def page_analytics(student) -> None:
    from modules.analytics import (
        detect_learning_profile,
        learning_health_score,
        exam_performance_band
    )
    
    st.markdown("## Intelligence Analytics")
    st.caption("Deep insights into your learning behavior and academic performance.")

    # --- TOP ROW: CORE METRICS ---
    score = learning_score(student)
    pred_score = predict_exam_score(student)
    health = learning_health_score(student)
    cons = study_consistency_score(student)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Score", f"{pred_score:.0f}/100", help="Predicted exam result based on current performance.")
    c2.metric("Learning Score", f"{score:.1f}", help="Overall AI intelligence metric.")
    c3.metric("Health Score", f"{health:.0f}%", help="Balanced score of accuracy, time, and consistency.")
    c4.metric("Consistency", f"{cons:.0f}%", help="Percentage of active study days in the last week.")

    st.divider()

    # --- SECOND ROW: PROFILE & HEALTH ---
    col_profile, col_health = st.columns([0.6, 0.4])
    
    with col_profile:
        profile, desc, rec = detect_learning_profile(student)
        st.markdown(f"### 🎖️ Learning Profile: {profile}")
        with st.container(border=True):
            st.markdown(f"**Description:** {desc}")
            st.info(f"💡 **Recommendation:** {rec}")
            
            # Progress to next level logic (visual only)
            st.caption("Progress to Elite Status")
            st.progress(min(1.0, score / 100))

    with col_health:
        st.markdown("### 🏥 Learning Health")
        with st.container(border=True):
            st.progress(health / 100)
            if health > 80:
                st.success(f"Health: {health}% - Optimized")
            elif health > 50:
                st.warning(f"Health: {health}% - Stable")
            else:
                st.error(f"Health: {health}% - Needs Attention")
            st.caption("A balanced measure of accuracy and daily habits.")

    # --- THIRD ROW: ADVANCED INSIGHTS & CHARTS ---
    st.markdown("### 📊 Performance Insights")
    df = build_analytics_frame(student)
    
    # Identify strongest/weakest for insights
    valid_metrics = df[~df["Metric"].isin(["Engagement Score", "Learning Score", "Consistency Score", "Predicted Exam Score"])]
    strongest = valid_metrics.loc[valid_metrics["Value"].idxmax()]
    weakest = valid_metrics.loc[valid_metrics["Value"].idxmin()]

    col_chart, col_cards = st.columns([0.65, 0.35])
    
    with col_chart:
        chart_data = df[df["Metric"].isin(["Quiz Accuracy (%)", "Engagement Score", "Learning Score", "Consistency Score"])].set_index("Metric")
        st.bar_chart(chart_data)

    with col_cards:
        st.markdown(f"""
        <div class="ald-card">
            <h4>🌟 Strongest Metric</h4>
            <div class="ald-muted">{strongest['Metric']}</div>
            <div style="font-size: 1.2rem; color: #50fa7b; font-weight: bold;">{strongest['Value']}</div>
        </div>
        <div class="ald-card" style="margin-top: 10px;">
            <h4>📉 Area to Improve</h4>
            <div class="ald-muted">{weakest['Metric']}</div>
            <div style="font-size: 1.2rem; color: #ff5555; font-weight: bold;">{weakest['Value']}</div>
        </div>
        <div class="ald-card" style="margin-top: 10px;">
            <h4>🎯 Predicted Band</h4>
            <div style="font-size: 1.2rem; color: #8be9fd; font-weight: bold;">{exam_performance_band(pred_score)}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- FOURTH ROW: TOPIC MASTERY & STREAKS ---
    st.divider()
    col_topic, col_cons = st.columns([0.6, 0.4])
    
    with col_topic:
        st.markdown("### 🎓 Subject Mastery")
        mastery = topic_mastery(student)
        topic_df = pd.DataFrame([{"Subject": k, "Mastery": v} for k, v in mastery.items()])
        st.bar_chart(topic_df, x="Subject", y="Mastery")
        
        weak_topics = [k for k, v in mastery.items() if v < 60]
        strong_topics = [k for k, v in mastery.items() if v >= 85]
        
        if weak_topics:
            st.warning(f"Focus required on: **{', '.join(weak_topics)}**")
        if strong_topics:
            st.success(f"Excellent mastery in: **{', '.join(strong_topics)}**")

    with col_cons:
        st.markdown("### 🔥 Consistency & Habits")
        with st.container(border=True):
            # Simulation of activity streak
            streak = 0
            if student.daily_activity:
                sorted_days = sorted(student.daily_activity.keys(), reverse=True)
                import datetime
                today = datetime.date.today()
                for i, day_str in enumerate(sorted_days):
                    d = datetime.date.fromisoformat(day_str)
                    if d == today - datetime.timedelta(days=i):
                        streak += 1
                    else:
                        break
            
            st.markdown(f"#### Current Streak: {streak} Days")
            st.caption("Consistency is key to long-term retention.")
            
            if cons > 75:
                st.write("✅ You are a daily learner! Keep it up.")
            elif cons > 40:
                st.write("⚠️ Try to log in for at least 5 mins every day.")
            else:
                st.write("🚨 Frequent breaks detected. Start a 3-day streak to reset!")


def page_profile(student) -> None:
    st.markdown("## Student Profile")
    st.caption("Profile card + key learning stats.")

    col1, col2 = st.columns([0.35, 0.65], vertical_alignment="top")
    with col1:
        with st.container(border=True):
            st.markdown(f"### {student.name}")
            st.write(student.email)
            st.divider()
            st.metric("Total XP", student.xp)
            st.metric("Lessons completed", student.lessons_completed)
            st.metric("Quizzes taken", student.quizzes_attempted)
            st.metric("Videos watched", student.videos_watched)
            st.metric("AI questions", student.ai_questions_asked)

    with col2:
        st.markdown("### Learning progress")
        score = learning_score(student)
        st.progress(min(1.0, score / 100.0))
        st.write(f"Learning Score: **{score:.2f}** ({detect_strength_band(score)})")

        if student.favorite_summaries:
            st.divider()
            st.markdown("### ⭐ Favorite Summaries")
            for i, fav in enumerate(reversed(student.favorite_summaries)):
                with st.expander(f"{fav['subject']} - {fav['chapter']}", expanded=False):
                    st.markdown(fav["summary"])
                    if st.button("Remove", key=f"remove_fav_{i}"):
                        student.favorite_summaries.remove(fav)
                        save_student(student)
                        st.rerun()


def page_leaderboard(student) -> None:
    st.markdown("## Leaderboard")
    st.caption("Points from videos, quizzes, AI tutor usage, lessons, and XP.")

    df = leaderboard_df()
    st.dataframe(df, use_container_width=True, hide_index=True)

    if not df.empty:
        me = df[df["Email"] == student.email]
        if not me.empty:
            st.info(f"Your rank: **{int(me.iloc[0]['Rank'])}** | Score: **{int(me.iloc[0]['Score'])}**")


def main() -> None:
    production_setup()
    apply_theme()
    inject_css()

    if not st.session_state.get("logged_in"):
        if st.session_state.get("auth_page") == "signup":
            page_signup()
        else:
            page_login()
        return

    sidebar_profile()
    nav = sidebar_nav()

    student = get_active_student()

    # Render Sidebar Footer (User Info Section) at the bottom
    with st.sidebar:
        st.markdown("<br>" * 3, unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-user-card">
                <div style="font-size: 0.9rem; font-weight: 700; color: #8be9fd;">👤 {student.name}</div>
                <div style="font-size: 0.75rem; opacity: 0.8;">{student.email}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Logout", key="logout_btn", width="stretch"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if nav == "home":
        page_home(student)
    elif nav == "learning":
        page_learning(student)
    elif nav == "tutor":
        page_ai_tutor(student)
    elif nav == "quiz":
        page_quiz(student)
    elif nav == "videos":
        page_videos(student)
    elif nav == "notes":
        page_notes(student)
    elif nav == "analytics":
        page_analytics(student)
    elif nav == "profile":
        page_profile(student)
    elif nav == "leaderboard":
        page_leaderboard(student)
    else:
        st.error("Unknown page.")


if __name__ == "__main__":
    main()