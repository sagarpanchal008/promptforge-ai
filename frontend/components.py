"""
frontend/components.py
Member 5: Frontend & Integration Engineer
-----------------------------------------
Reusable AI SaaS UI components for Intelligent Prompt Optimization Assistant 2.0:
KPI cards, status badges, quality dimension bars, Plotly layout styling, and
empty-state callouts.
"""

from __future__ import annotations

import html
from typing import Any, Dict, Optional

import plotly.graph_objects as go
import streamlit as st

from frontend.styles import PALETTE


def render_page_header(title: str, subtitle: str, badge_text: str = "v2.0 Platform") -> None:
    """Render a clean SaaS top banner with title, subtitle, and status pill."""
    safe_title = html.escape(title)
    safe_sub = html.escape(subtitle)
    safe_badge = html.escape(badge_text)
    st.markdown(
        f"""
        <div class="saas-card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem;">
            <div>
                <h2 style="margin:0; font-size:1.45rem;">{safe_title}</h2>
                <p class="muted-text" style="margin:0.25rem 0 0 0;">{safe_sub}</p>
            </div>
            <span class="badge badge-primary">{safe_badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    subtitle: str = "",
    accent_color: str = "#2788F5",
) -> None:
    """Render a rounded KPI metric card with subtle top accent border."""
    safe_label = html.escape(label)
    safe_val = html.escape(value)
    safe_sub = html.escape(subtitle)
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 3px solid {accent_color};">
            <div class="kpi-label">{safe_label}</div>
            <div class="kpi-value">{safe_val}</div>
            <div class="kpi-sub">{safe_sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(risk_level: str) -> str:
    """Return an HTML status pill for Low, Medium, or High risk / quality labels."""
    level = (risk_level or "Low").strip()
    mapping = {
        "Low": "badge-success",
        "Good": "badge-success",
        "Medium": "badge-warning",
        "Average": "badge-warning",
        "High": "badge-error",
        "Poor": "badge-error",
    }
    cls = mapping.get(level, "badge-primary")
    return f'<span class="badge {cls}">{html.escape(level)}</span>'


def render_dimension_bars(dimensions: Dict[str, float]) -> None:
    """Render horizontal progress bars for Clarity, Specificity, Context, and Completeness."""
    if not dimensions:
        return

    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("**Quality Dimension Breakdown**")
    for key, val in dimensions.items():
        score = max(0.0, min(100.0, float(val)))
        color = (
            PALETTE["success"]
            if score >= 75
            else (PALETTE["warning"] if score >= 50 else PALETTE["error"])
        )
        label = html.escape(key.replace("_", " ").title())
        st.markdown(
            f"""
            <div style="margin-bottom:0.65rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.86rem; margin-bottom:0.2rem;">
                    <span>{label}</span>
                    <span style="font-weight:600; color:{color};">{score:.1f} / 100</span>
                </div>
                <div style="width:100%; height:8px; background:#101827; border-radius:999px; overflow:hidden;">
                    <div style="width:{score}%; height:100%; background:{color}; border-radius:999px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_empty_state(title: str, description: str) -> None:
    """Render a clean empty-state card when no stored data exists yet."""
    st.markdown(
        f"""
        <div class="saas-card" style="text-align:center; padding:2.2rem 1.5rem; border-style:dashed;">
            <h4 style="margin:0 0 0.4rem 0; color:#F8FAFC;">{html.escape(title)}</h4>
            <p class="muted-text" style="margin:0 auto; max-width:520px;">{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_saas_theme(fig: go.Figure, height: int = 340, title: Optional[str] = None) -> go.Figure:
    """Apply consistent dark SaaS styling and responsive sizing to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PALETTE["text_primary"])) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16,24,39,0.55)",
        font=dict(family="Inter, sans-serif", color=PALETTE["text_primary"], size=12),
        height=height,
        margin=dict(l=24, r=24, t=44 if title else 24, b=28),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["text_secondary"])),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.18)")
    return fig