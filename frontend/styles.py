"""
frontend/styles.py
Member 5: Frontend & Integration Engineer
-----------------------------------------
Professional AI SaaS Dark-Mode Theme & Responsive CSS Engine.
Implements the exact v2.0 color palette and responsive media queries across
1920px desktop, 1366px laptop, 768px tablet, and 390px smartphone viewports
without breaking native Streamlit widgets or sidebar drawer behavior.
"""

from __future__ import annotations

import streamlit as st

PALETTE = {
    "bg_primary": "#101827",
    "bg_secondary": "#17243A",
    "bg_card": "#1D2B42",
    "accent_primary": "#2788F5",
    "accent_secondary": "#8B5CF6",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
}


def inject_saas_styles() -> None:
    """Inject responsive SaaS CSS into the Streamlit application."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg-primary: #101827;
            --bg-secondary: #17243A;
            --bg-card: #1D2B42;
            --accent-primary: #2788F5;
            --accent-secondary: #8B5CF6;
            --success: #22C55E;
            --warning: #F59E0B;
            --error: #EF4444;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --border-subtle: rgba(148, 163, 184, 0.16);
        }

        /* Global App Canvas */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        [data-testid="stHeader"] {
            background-color: rgba(16, 24, 39, 0.88) !important;
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--border-subtle);
        }

        /* Sidebar Navigation */
        [data-testid="stSidebar"] {
            background-color: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-subtle) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text-primary);
        }

        /* Main Container Spacing */
        .block-container {
            max-width: 1440px !important;
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
        }

        /* Typography */
        h1, h2, h3, h4 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.015em;
        }

        p, span, label, .stMarkdown {
            color: var(--text-primary);
        }

        .muted-text {
            color: var(--text-secondary) !important;
            font-size: 0.9rem;
        }

        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* SaaS Cards */
        .saas-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.22);
            margin-bottom: 1rem;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }

        .saas-card:hover {
            border-color: rgba(39, 136, 245, 0.45);
        }

        .kpi-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18);
            height: 100%;
        }

        .kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary) !important;
            margin-bottom: 0.35rem;
        }

        .kpi-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary) !important;
            line-height: 1.2;
        }

        .kpi-sub {
            font-size: 0.82rem;
            color: var(--text-secondary) !important;
            margin-top: 0.35rem;
        }

        /* Status Badges */
        .badge {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .badge-primary { background: rgba(39, 136, 245, 0.18); color: #60A5FA !important; border: 1px solid rgba(39, 136, 245, 0.4); }
        .badge-purple  { background: rgba(139, 92, 246, 0.18); color: #C4B5FD !important; border: 1px solid rgba(139, 92, 246, 0.4); }
        .badge-success { background: rgba(34, 197, 94, 0.16); color: #4ADE80 !important; border: 1px solid rgba(34, 197, 94, 0.4); }
        .badge-warning { background: rgba(245, 158, 11, 0.16); color: #FBBF24 !important; border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-error   { background: rgba(239, 68, 68, 0.16); color: #F87171 !important; border: 1px solid rgba(239, 68, 68, 0.4); }

        /* Inputs, TextAreas & Selectboxes */
        .stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 10px !important;
        }

        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 0 2px rgba(39, 136, 245, 0.22) !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 9px !important;
            font-weight: 600 !important;
            min-height: 2.65rem !important;
            transition: all 0.18s ease !important;
            border: 1px solid var(--border-subtle) !important;
        }

        .stButton > button[kind="primary"] {
            background-color: var(--accent-primary) !important;
            color: #FFFFFF !important;
            border: none !important;
        }

        .stButton > button[kind="primary"]:hover {
            background-color: #1D74D8 !important;
            transform: translateY(-1px);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background-color: var(--bg-secondary);
            padding: 0.35rem;
            border-radius: 10px;
            border: 1px solid var(--border-subtle);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: var(--text-secondary) !important;
            font-weight: 600;
            padding: 0.45rem 0.9rem;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }

        /* Prevent Horizontal Overflow */
        .stPlotlyChart, [data-testid="stDataFrame"], table {
            max-width: 100% !important;
            overflow-x: auto !important;
        }

        /* -------------------------------------------------------------
           Responsive Media Queries (1920px, 1366px, 768px, 390px)
           ------------------------------------------------------------- */
        @media (max-width: 1366px) {
            .block-container {
                padding-left: 1.5rem !important;
                padding-right: 1.5rem !important;
            }
            .kpi-value {
                font-size: 1.5rem;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 1rem !important;
            }
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
            .saas-card, .kpi-card {
                padding: 1rem;
            }
            .kpi-value {
                font-size: 1.35rem;
            }
        }

        @media (max-width: 480px) {
            .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
            }
            h1 {
                font-size: 1.45rem !important;
            }
            h2 {
                font-size: 1.2rem !important;
            }
            .stButton > button {
                width: 100% !important;
                min-height: 2.85rem !important;
            }
            .stTextArea textarea {
                font-size: 0.92rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )