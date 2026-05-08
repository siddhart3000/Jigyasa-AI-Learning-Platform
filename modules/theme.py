from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """
    Apply the global Dark Neon theme using CSS variables.
    """
    bg = "#0a0a0f"
    card = "#111122"
    text = "#ffffff"
    muted = "rgba(255,255,255,0.72)"
    accent = "#00eaff"
    accent2 = "#8a2be2"
    accent3 = "#00ffcc"
    ring = "#00ffcc"

    st.markdown(
        f"""
        <style>
          :root {{
            --bg: {bg};
            --card: {card};
            --text: {text};
            --muted: {muted};
            --accent: {accent};
            --accent2: {accent2};
            --accent3: {accent3};
            --ring: {ring};
          }}

          html, body, [data-testid="stAppViewContainer"] {{
            background: var(--bg) !important;
            color: var(--text) !important;
          }}

          /* Sidebar */
          [data-testid="stSidebar"] {{
            background: color-mix(in srgb, var(--bg) 88%, #000 12%) !important;
            border-right: 1px solid color-mix(in srgb, var(--text) 10%, transparent) !important;
          }}

          /* Modern card */
          .ald-card, [data-testid="stMetric"], .st-emotion-cache-1r6slb0 {{
            background: color-mix(in srgb, var(--card) 92%, transparent) !important;
            border: 1px solid color-mix(in srgb, var(--text) 12%, transparent) !important;
            border-radius: 14px !important;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
          }}
          .ald-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 30px color-mix(in srgb, var(--accent) 20%, transparent);
            border-color: color-mix(in srgb, var(--accent) 35%, transparent) !important;
          }}

          /* Buttons */
          .stButton>button {{
            border-radius: 12px !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
          }}
          .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 0 14px color-mix(in srgb, var(--accent) 55%, transparent);
          }}

          /* Primary button contrast in light mode */
          .stButton>button[kind="primary"] {{
            background: var(--accent) !important;
            color: #fff !important;
            border: 1px solid color-mix(in srgb, var(--accent) 75%, #000 25%) !important;
          }}

          /* Hero gradient */
          .ald-hero-title {{
            background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
            -webkit-background-clip: text !important;
            color: transparent !important;
          }}
          .ald-hero-sub {{ color: var(--muted) !important; }}
          .ald-muted {{ color: var(--muted) !important; }}

          /* Progress ring */
          .ald-ring {{
            background: conic-gradient(var(--ring) calc(var(--val)*1%), color-mix(in srgb, var(--ring) 12%, transparent) 0) !important;
            box-shadow: 0 0 18px color-mix(in srgb, var(--ring) 55%, transparent) !important;
          }}

          /* Smooth transitions */
          * {{ transition: background-color 0.25s ease, color 0.25s ease; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
