from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class NavItem:
    key: str
    label: str
    icon: str


NAV_ITEMS = [
    NavItem("home", "Home", "🏠"),
    NavItem("learning", "Learning", "📚"),
    NavItem("tutor", "AI Tutor", "🤖"),
    NavItem("quiz", "Quiz", "📝"),
    NavItem("analytics", "Analytics", "📊"),
    NavItem("leaderboard", "Leaderboard", "🏆"),
    NavItem("profile", "Profile", "👤"),
]


def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown("### 🎓 Jigyasa\nAI Learning Platform")
        labels = [f"{i.icon}  {i.label}" for i in NAV_ITEMS]
        current = st.session_state.get("nav_label", labels[0])
        choice = st.radio("Navigation", options=labels, index=labels.index(current) if current in labels else 0)
        st.session_state["nav_label"] = choice
        key = next(i.key for i in NAV_ITEMS if f"{i.icon}  {i.label}" == choice)
    return key


def card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="border:1px solid rgba(255,255,255,0.12); border-radius:14px; padding:16px; background:rgba(255,255,255,0.03);">
            <div style="font-size:14px; opacity:0.8;">{title}</div>
            <div style="font-size:22px; font-weight:700; margin-top:6px;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tile(title: str, subtitle: str, icon: str = "📘") -> None:
    st.markdown(
        f"""
        <div class="ald-card ald-tile">
          <h4>{icon} {title}</h4>
          <div class="ald-muted" style="margin-top:6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

