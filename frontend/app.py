"""
frontend/app.py
Member 5: Frontend & Integration Engineer
-----------------------------------------
Intelligent Prompt Optimization Assistant 2.0
AI-Powered Prompt Intelligence, Evaluation, Security & Optimization Platform.
Integrates Member 1 (Deep Learning), Member 2 (LangGraph Multi-Agent),
Member 3 (Hybrid Security & A/B Playground), and Member 4 (SQLite Store &
Version Control) inside a responsive 7-view SaaS interface.
"""

from __future__ import annotations

import difflib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path when launched directly via `streamlit run frontend/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from agents.deep_learning_agent import evaluate_prompt_intelligence
from agents.workflow import run_optimization_pipeline
from database.models import FeedbackRecord, PromptRecord
from database.prompt_store import PromptStore
from frontend.components import (
    apply_plotly_saas_theme,
    render_dimension_bars,
    render_empty_state,
    render_kpi_card,
    render_page_header,
    risk_badge_html,
)
from frontend.styles import PALETTE, inject_saas_styles
from playground.ab_testing import generate_prompt_variants, run_ab_experiment
from playground.providers import list_providers
from security.scanner import scan_prompt_security
from utils.evaluation import load_saved_benchmark_results

load_dotenv(override=True)

st.set_page_config(
    page_title="PromptForge AI 2.0 | Prompt Intelligence & Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_saas_styles()


@st.cache_resource
def get_store() -> PromptStore:
    return PromptStore()


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_benchmarks() -> Dict[str, Any]:
    return load_saved_benchmark_results()


@st.cache_data(ttl=120, show_spinner=False)
def cached_dl_eval(prompt_text: str) -> Dict[str, Any]:
    return evaluate_prompt_intelligence(prompt_text)


store = get_store()

NAV_PAGES = [
    "Overview Dashboard",
    "Prompt Optimizer",
    "AI Playground",
    "Deep Learning Benchmark",
    "Security Scanner",
    "Prompt Library",
    "Analytics",
]


# ---------------------------------------------------------------------------
# Sidebar Navigation & System Status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.4rem 0 0.8rem 0;">
            <div style="font-size:1.18rem; font-weight:700; color:#F8FAFC;">⚡ PromptForge AI</div>
            <div style="font-size:0.78rem; color:#94A3B8;">Prompt Intelligence Platform v2.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "Navigation",
        NAV_PAGES,
        label_visibility="collapsed",
    )

    st.markdown("---")
    gemini_ready = bool(os.getenv("GEMINI_API_KEY", "").strip())
    status_badge = (
        '<span class="badge badge-success">Gemini API Connected</span>'
        if gemini_ready
        else '<span class="badge badge-warning">GEMINI_API_KEY Missing</span>'
    )
    st.markdown(
        f"""
        <div class="saas-card" style="padding:0.85rem;">
            <div style="font-size:0.76rem; color:#94A3B8; margin-bottom:0.35rem;">ENGINE STATUS</div>
            {status_badge}
            <div style="font-size:0.78rem; color:#94A3B8; margin-top:0.55rem;">
                Models: <b>1D-CNN + Bi-LSTM</b><br/>
                Orchestrator: <b>LangGraph (7 Agents)</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ===========================================================================

def page_overview() -> None:
    render_page_header(
        "Overview Dashboard",
        "Real-time prompt intelligence, multi-agent optimization performance, and security posture.",
        badge_text="Live Telemetry",
    )

    analytics = store.get_dashboard_analytics()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Prompts Analyzed",
            str(analytics["total_analyzed"]),
            f"{analytics['total_saved']} saved in library",
            PALETTE["accent_primary"],
        )
    with c2:
        render_kpi_card(
            "Average Quality Score",
            f"{analytics['avg_quality']:.1f} / 100",
            f"Avg gain: {analytics['avg_improvement']:+.1f} pts",
            PALETTE["success"],
        )
    with c3:
        render_kpi_card(
            "Optimization Success",
            f"{analytics['success_rate']:.1f}%",
            "Prompts with positive quality delta",
            PALETTE["accent_secondary"],
        )
    with c4:
        render_kpi_card(
            "Security Warnings",
            str(analytics["security_warnings"]),
            "Medium & High risk detections",
            PALETTE["warning"],
        )

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("#### Recent Optimization Activity")
        recent = analytics.get("recent_activity", [])
        if not recent:
            render_empty_state(
                "No Optimization Activity Yet",
                "Run your first prompt through the Prompt Optimizer to populate live quality gains and history.",
            )
        else:
            df_recent = pd.DataFrame(recent)
            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    name="Original Score",
                    x=df_recent["title"],
                    y=df_recent["original_score"],
                    marker_color="#64748B",
                )
            )
            fig_bar.add_trace(
                go.Bar(
                    name="Optimized Score",
                    x=df_recent["title"],
                    y=df_recent["optimized_score"],
                    marker_color=PALETTE["accent_primary"],
                )
            )
            fig_bar.update_layout(barmode="group")
            apply_plotly_saas_theme(fig_bar, height=330, title="Before vs. After Optimization Quality")
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("#### Prompt Categories")
        cat_dist = analytics.get("category_distribution", {})
        if not cat_dist:
            render_empty_state(
                "No Categorized Prompts",
                "Categories appear automatically as prompts are analyzed and stored.",
            )
        else:
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=list(cat_dist.keys()),
                        values=list(cat_dist.values()),
                        hole=0.52,
                        marker=dict(
                            colors=[
                                PALETTE["accent_primary"],
                                PALETTE["accent_secondary"],
                                PALETTE["success"],
                                PALETTE["warning"],
                                "#38BDF8",
                            ]
                        ),
                    )
                ]
            )
            apply_plotly_saas_theme(fig_pie, height=330, title="Workload Mix by Domain")
            st.plotly_chart(fig_pie, use_container_width=True)

    if analytics.get("recent_activity"):
        st.markdown("#### Latest Saved Revisions")
        df_table = pd.DataFrame(analytics["recent_activity"])[
            ["id", "title", "category", "original_score", "optimized_score", "delta", "security_risk"]
        ]
        df_table.columns = ["ID", "Title", "Category", "Original", "Optimized", "Gain", "Security"]
        st.dataframe(df_table, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 2: PROMPT OPTIMIZER (DL ENGINE + 7-AGENT LANGGRAPH WORKFLOW)
# ===========================================================================

def page_optimizer() -> None:
    render_page_header(
        "Prompt Optimizer & Intelligence Engine",
        "Deep Learning quality diagnosis (1D-CNN + Bi-LSTM) paired with a 7-agent autonomous LangGraph optimization loop.",
        badge_text="7-Agent LangGraph",
    )

    collections = store.list_collections()
    col_map = {f"{c['name']} (#{c['id']})": c["id"] for c in collections}

    raw_prompt = st.text_area(
        "Enter your prompt to analyze, secure, and optimize",
        height=160,
        placeholder="Example: Write a python script to process customer sales data and find top products.",
    )

    cfg1, cfg2, cfg3, cfg4 = st.columns([2, 1, 2, 1])
    with cfg1:
        target_model = st.selectbox(
            "Target Model",
            ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3.8-flash", "General LLM"],
        )
    with cfg2:
        max_iter = st.slider("Max Agent Cycles", min_value=1, max_value=4, value=2)
    with cfg3:
        selected_col_label = st.selectbox("Save to Collection", list(col_map.keys()))
    with cfg4:
        save_to_lib = st.checkbox("Auto-Save", value=True)

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        analyze_only_btn = st.button("🔍 Deep Learning Diagnostic Only", use_container_width=True)
    with btn_col2:
        optimize_btn = st.button("⚡ Run Autonomous Multi-Agent Optimization", type="primary", use_container_width=True)

    if analyze_only_btn:
        if not raw_prompt.strip():
            st.warning("Enter a prompt first to run Deep Learning analysis.")
        else:
            with st.spinner("Running 1D-CNN, Bi-LSTM, and dimension analysis..."):
                dl_res = cached_dl_eval(raw_prompt)
                sec_res = scan_prompt_security(raw_prompt, use_ai=False)
                store.log_activity(
                    event_type="analyze",
                    original_score=dl_res["overall_score"],
                    security_risk=sec_res["risk_level"],
                )
                st.session_state["last_dl_diag"] = dl_res
                st.session_state["last_sec_diag"] = sec_res
                st.session_state.pop("last_opt_result", None)

    if optimize_btn:
        if not raw_prompt.strip():
            st.warning("Enter a prompt first before launching the multi-agent pipeline.")
        else:
            with st.spinner("Executing 7-Agent LangGraph Pipeline (Analyze → DL Score → Security → Strategy → Optimize → Review)..."):
                try:
                    result = run_optimization_pipeline(
                        raw_prompt=raw_prompt,
                        target_model=target_model,
                        max_iterations=max_iter,
                    )
                    saved_id = None
                    if save_to_lib:
                        rec = PromptRecord(
                            original_prompt=result["original_prompt"],
                            optimized_prompt=result["optimized_prompt"],
                            category=result.get("task_category", "General"),
                            target_model=target_model,
                            collection_id=col_map[selected_col_label],
                            original_score=result.get("original_score", 0.0),
                            optimized_score=result.get("optimized_score", 0.0),
                            cnn_prediction=result.get("cnn_prediction", {}).get("label", "Unknown"),
                            bilstm_prediction=result.get("bilstm_prediction", {}).get("label", "Unknown"),
                            dimension_scores=result.get("dimension_scores", {}),
                            weaknesses=result.get("weaknesses", []),
                            strategy_applied=result.get("strategy_used", ""),
                            security_risk=result.get("security_report", {}).get("risk_level", "Low"),
                            optimization_settings={"target_model": target_model, "max_iterations": max_iter},
                        )
                        saved = store.save_prompt(rec)
                        saved_id = saved.id

                    store.log_activity(
                        event_type="optimize",
                        category=result.get("task_category", "General"),
                        original_score=result.get("original_score", 0.0),
                        optimized_score=result.get("optimized_score", 0.0),
                        security_risk=result.get("security_report", {}).get("risk_level", "Low"),
                    )
                    result["saved_prompt_id"] = saved_id
                    st.session_state["last_opt_result"] = result
                except Exception as exc:
                    st.error(f"Optimization pipeline error: {exc}")

    # Render Deep Learning Diagnostic if triggered
    if "last_dl_diag" in st.session_state and "last_opt_result" not in st.session_state:
        diag = st.session_state["last_dl_diag"]
        st.markdown("### Deep Learning Prompt Intelligence Report")
        m1, m2, m3 = st.columns(3)
        with m1:
            render_kpi_card("Overall Quality", f"{diag['overall_score']:.1f} / 100", "Blended DL + Structural Score")
        with m2:
            cnn = diag["cnn_prediction"]
            render_kpi_card("1D-CNN Prediction", cnn["label"], f"Confidence: {cnn['confidence']:.1f}%", PALETTE["accent_secondary"])
        with m3:
            lstm = diag["bilstm_prediction"]
            render_kpi_card("Bi-LSTM Prediction", lstm["label"], f"Confidence: {lstm['confidence']:.1f}%", PALETTE["success"])

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            render_dimension_bars(diag["dimension_scores"])
        with d_col2:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("**Detected Weaknesses & Missing Instructions**")
            for w in diag["weaknesses"]:
                st.markdown(f"- {w}")
            st.markdown("**Actionable Recommendations**")
            for r in diag["recommendations"]:
                st.markdown(f"- {r}")
            st.markdown("</div>", unsafe_allow_html=True)

    # Render Full Multi-Agent Optimization Result
    if "last_opt_result" in st.session_state:
        res = st.session_state["last_opt_result"]
        st.markdown("---")
        st.success(res.get("explanation", "Optimization completed."))

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card(
                "Quality Transformation",
                f"{res['original_score']:.1f} → {res['optimized_score']:.1f}",
                f"Net Gain: {res['improvement_delta']:+.1f} pts",
                PALETTE["success"],
            )
        with k2:
            cnn = res.get("cnn_prediction", {})
            render_kpi_card(
                "1D-CNN Assessment",
                str(cnn.get("label", "N/A")),
                f"Confidence: {cnn.get('confidence', 0):.1f}%",
                PALETTE["accent_primary"],
            )
        with k3:
            lstm = res.get("bilstm_prediction", {})
            render_kpi_card(
                "Bi-LSTM Assessment",
                str(lstm.get("label", "N/A")),
                f"Confidence: {lstm.get('confidence', 0):.1f}%",
                PALETTE["accent_secondary"],
            )
        with k4:
            sec_lvl = res.get("security_report", {}).get("risk_level", "Low")
            render_kpi_card(
                "Security Posture",
                sec_lvl,
                f"Cycles: {res.get('iterations_completed', 1)} | {res.get('strategy_used', '')}",
                PALETTE["warning"] if sec_lvl != "Low" else PALETTE["success"],
            )

        comp_left, comp_right = st.columns(2)
        with comp_left:
            st.markdown("#### Original Prompt")
            st.text_area("Original", value=res["original_prompt"], height=260, disabled=True, label_visibility="collapsed")
        with comp_right:
            st.markdown("#### Optimized Prompt (Copy-Ready)")
            st.text_area("Optimized", value=res["optimized_prompt"], height=260, label_visibility="collapsed")

        det1, det2 = st.columns(2)
        with det1:
            render_dimension_bars(res.get("dimension_scores", {}))
        with det2:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown(f"**Task Category:** `{res.get('task_category', 'General')}`")
            st.markdown(f"**Strategy Applied:** `{res.get('strategy_used', 'N/A')}`")
            st.markdown("**Detected Weaknesses Addressed:**")
            for w in res.get("weaknesses", []):
                st.markdown(f"- {w}")
            if res.get("missing_instructions"):
                st.markdown("**Missing Instructions Filled:**")
                for m in res["missing_instructions"]:
                    st.markdown(f"- {m}")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🤖 Autonomous Multi-Agent Execution Trace (7 Agents)", expanded=False):
            for step in res.get("agent_trace", []):
                st.markdown(
                    f"**{step.get('agent')}** — `{step.get('status')}`: {step.get('summary')}"
                )

        # Module 6: Feedback-Based Optimization Capture
        st.markdown("#### Rate This Optimization")
        with st.form("optimizer_feedback_form"):
            fb_c1, fb_c2, fb_c3 = st.columns([1, 1, 3])
            with fb_c1:
                vote = st.radio("Outcome", ["👍 Helpful", "👎 Needs Work"], horizontal=True)
            with fb_c2:
                stars = st.slider("Quality Rating", 1, 5, 5)
            with fb_c3:
                fb_comment = st.text_input("Optional Notes", placeholder="What worked well or could be improved?")
            submitted_fb = st.form_submit_button("Submit Feedback")
            if submitted_fb:
                store.save_feedback(
                    FeedbackRecord(
                        prompt_id=res.get("saved_prompt_id"),
                        original_prompt=res["original_prompt"],
                        optimized_prompt=res["optimized_prompt"],
                        thumbs_up=vote.startswith("👍"),
                        rating=stars,
                        comment=fb_comment,
                        detected_weaknesses=res.get("weaknesses", []),
                        strategy_used=res.get("strategy_used", ""),
                    )
                )
                st.success("Feedback saved persistently for analytics and future dataset curation.")


# ===========================================================================
# PAGE 3: AI PLAYGROUND (A/B PROMPT TESTING)
# ===========================================================================

def page_playground() -> None:
    render_page_header(
        "AI Playground & Prompt A/B Testing",
        "Generate multiple engineered prompt variants, execute live LLM calls, and benchmark output quality.",
        badge_text="Live Multi-Variant Eval",
    )

    providers = list_providers()
    provider_labels = {f"{p['name']} ({'Ready' if p['configured'] else 'Key Missing'})": p["id"] for p in providers}

    base_prompt = st.text_area(
        "Base Prompt for A/B Experiment",
        height=130,
        placeholder="Enter a prompt to generate competing variants and test live LLM responses...",
    )

    p1, p2, p3 = st.columns([2, 1, 1])
    with p1:
        chosen_prov_label = st.selectbox("LLM Provider", list(provider_labels.keys()))
        chosen_provider = provider_labels[chosen_prov_label]
    with p2:
        num_variants = st.selectbox("Variants to Generate", [2, 3, 4], index=1)
    with p3:
        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
        gen_btn = st.button("✨ Generate Variants", use_container_width=True)

    if gen_btn:
        if not base_prompt.strip():
            st.warning("Enter a base prompt first.")
        else:
            with st.spinner("Generating structured prompt engineering variants..."):
                try:
                    variants = generate_prompt_variants(base_prompt, num_variants=num_variants)
                    st.session_state["playground_variants"] = variants
                    st.session_state["playground_base"] = base_prompt
                except Exception as exc:
                    st.error(f"Could not generate variants: {exc}")

    variants: List[Dict[str, str]] = st.session_state.get("playground_variants", [])
    if variants:
        st.markdown("#### Select Variants to Execute & Compare")
        selected_variants: List[Dict[str, str]] = []
        cols = st.columns(min(len(variants), 2))
        for idx, v in enumerate(variants):
            with cols[idx % 2]:
                st.markdown('<div class="saas-card">', unsafe_allow_html=True)
                include = st.checkbox(
                    f"{v['name']} — {v['strategy']}",
                    value=True,
                    key=f"ab_sel_{idx}",
                )
                edited_text = st.text_area(
                    f"Prompt Text ({v['name']})",
                    value=v["prompt_text"],
                    height=140,
                    key=f"ab_txt_{idx}",
                    label_visibility="collapsed",
                )
                st.markdown("</div>", unsafe_allow_html=True)
                if include:
                    selected_variants.append(
                        {
                            "name": v["name"],
                            "strategy": v["strategy"],
                            "prompt_text": edited_text,
                        }
                    )

        if st.button("🚀 Run Live A/B Comparison", type="primary"):
            if len(selected_variants) < 2:
                st.warning("Select at least 2 variants to compare.")
            else:
                with st.spinner("Sending live prompts to LLM and evaluating Relevance, Completeness & Instruction-Following..."):
                    try:
                        ab_out = run_ab_experiment(
                            base_prompt=st.session_state.get("playground_base", base_prompt),
                            variants=selected_variants,
                            provider_name=chosen_provider,
                        )
                        store.log_activity(event_type="ab_test")
                        st.session_state["ab_result"] = ab_out
                    except Exception as exc:
                        st.error(f"A/B experiment failed: {exc}")

    if "ab_result" in st.session_state:
        ab = st.session_state["ab_result"]
        win = ab["winner"]
        st.markdown(
            f"""
            <div class="saas-card" style="border-left: 4px solid #22C55E;">
                <span class="badge badge-success">WINNING VARIANT</span>
                <h3 style="margin:0.4rem 0 0.2rem 0;">🏆 {win['name']} ({win['strategy']}) — {win['composite_score']:.1f} / 100</h3>
                <p class="muted-text" style="margin:0;">{win['rationale']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Chart comparing Relevance, Completeness, Instruction-Following
        chart_rows = []
        for r in ab["results"]:
            m = r["metrics"]
            chart_rows.append(
                {
                    "Variant": r["name"],
                    "Relevance": m["relevance"],
                    "Completeness": m["completeness"],
                    "Instruction Following": m["instruction_following"],
                }
            )
        df_chart = pd.DataFrame(chart_rows)
        fig = go.Figure()
        for metric_col, color in [
            ("Relevance", PALETTE["accent_primary"]),
            ("Completeness", PALETTE["accent_secondary"]),
            ("Instruction Following", PALETTE["success"]),
        ]:
            fig.add_trace(
                go.Bar(name=metric_col, x=df_chart["Variant"], y=df_chart[metric_col], marker_color=color)
            )
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 105]))
        apply_plotly_saas_theme(fig, height=340, title="Live Response Evaluation Across Variants")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Side-by-Side Generated Responses")
        res_cols = st.columns(len(ab["results"]))
        for idx, r in enumerate(ab["results"]):
            with res_cols[idx]:
                m = r["metrics"]
                st.markdown(
                    f"""
                    <div class="saas-card">
                        <strong>{r['name']}</strong><br/>
                        <span class="muted-text">Score: <b>{m['composite_score']:.1f}</b> | {r['latency_ms']:.0f} ms | {r['word_count']} words</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if r["success"]:
                    st.text_area(
                        f"Output {idx}",
                        value=r["output_text"],
                        height=260,
                        label_visibility="collapsed",
                    )
                else:
                    st.error(r["error"])


# ===========================================================================
# PAGE 4: DEEP LEARNING BENCHMARK (1D-CNN vs BI-LSTM)
# ===========================================================================

def page_dl_benchmark() -> None:
    render_page_header(
        "Deep Learning Model Benchmark",
        "Architecture inspection and saved evaluation metrics for the 1D-CNN and Bidirectional LSTM classifiers.",
        badge_text="TensorFlow / Keras",
    )

    bench = get_cached_benchmarks()
    cnn_arch = bench["cnn_architecture"]
    bilstm_arch = bench["bilstm_architecture"]
    artifacts = bench["saved_artifacts"]

    c1, c2 = st.columns(2)
    with c1:
        render_kpi_card(
            "1D-CNN Model Artifact",
            cnn_arch["path"] or "Not Found",
            f"Trainable Parameters: {cnn_arch['total_params']:,}" if cnn_arch["available"] else "Place model in models/",
            PALETTE["accent_primary"],
        )
    with c2:
        render_kpi_card(
            "Bi-LSTM Model Artifact",
            bilstm_arch["path"] or "Not Found",
            f"Trainable Parameters: {bilstm_arch['total_params']:,}" if bilstm_arch["available"] else "Place model in models/",
            PALETTE["accent_secondary"],
        )

    # Layer-by-layer architecture tables
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("#### 1D-CNN Layer Architecture")
        if cnn_arch["available"] and cnn_arch["layers"]:
            st.dataframe(pd.DataFrame(cnn_arch["layers"]), use_container_width=True, hide_index=True)
        else:
            render_empty_state("1D-CNN Artifact Not Loaded", "Train or place your CNN .keras/.h5 model in models/ to inspect layers.")

    with arch_col2:
        st.markdown("#### Bi-LSTM Layer Architecture")
        if bilstm_arch["available"] and bilstm_arch["layers"]:
            st.dataframe(pd.DataFrame(bilstm_arch["layers"]), use_container_width=True, hide_index=True)
        else:
            render_empty_state("Bi-LSTM Artifact Not Loaded", "Train or place your Bi-LSTM .keras/.h5 model in models/ to inspect layers.")

    st.markdown("---")
    st.markdown("#### Saved Training & Evaluation Artifacts")
    if not artifacts:
        render_empty_state(
            "No Saved Evaluation JSON Artifacts Found",
            "Run notebooks/02_CNN_training.ipynb and notebooks/03_LSTM_training.ipynb to export evaluation metrics and confusion matrices into models/. No synthetic metrics are displayed.",
        )
    else:
        for name, payload in artifacts.items():
            if not isinstance(payload, dict):
                continue
            has_charts = any(
                k in payload for k in ("accuracy", "loss", "confusion_matrix")
            )
            if not has_charts:
                continue

            with st.expander(f"📊 {name.replace('_', ' ').title()}", expanded=True):
                if "accuracy" in payload and "val_accuracy" in payload:
                    epochs = list(range(1, len(payload["accuracy"]) + 1))
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(x=epochs, y=payload["accuracy"], mode="lines+markers", name="Train Accuracy"))
                    fig_acc.add_trace(go.Scatter(x=epochs, y=payload["val_accuracy"], mode="lines+markers", name="Val Accuracy"))
                    apply_plotly_saas_theme(fig_acc, height=300, title=f"{name} — Accuracy Curve")
                    st.plotly_chart(fig_acc, use_container_width=True)

                if "loss" in payload and "val_loss" in payload:
                    epochs = list(range(1, len(payload["loss"]) + 1))
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=epochs, y=payload["loss"], mode="lines+markers", name="Train Loss"))
                    fig_loss.add_trace(go.Scatter(x=epochs, y=payload["val_loss"], mode="lines+markers", name="Val Loss"))
                    apply_plotly_saas_theme(fig_loss, height=300, title=f"{name} — Loss Curve")
                    st.plotly_chart(fig_loss, use_container_width=True)

                if "confusion_matrix" in payload:
                    cm = payload["confusion_matrix"]
                    labels = payload.get("class_names", ["Poor", "Average", "Good"])
                    fig_cm = px.imshow(
                        cm,
                        x=labels,
                        y=labels,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted Label", y="True Label"),
                    )
                    apply_plotly_saas_theme(fig_cm, height=320, title=f"{name} — Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)


# ===========================================================================
# PAGE 5: SECURITY SCANNER
# ===========================================================================

def page_security_scanner() -> None:
    render_page_header(
        "Prompt Security Scanner",
        "Hybrid rule-based + AI threat auditing for prompt injection, jailbreaks, override directives, and PII/key exposure.",
        badge_text="Zero-Trust Input Audit",
    )

    untrusted_prompt = st.text_area(
        "Untrusted Prompt / External Content to Audit",
        height=160,
        placeholder="Paste a prompt or external user input to scan for injection, jailbreaks, or leaked credentials...",
    )

    s1, s2 = st.columns([2, 1])
    with s1:
        use_ai = st.checkbox("Include Gemini Semantic Threat Audit (in addition to deterministic rules)", value=True)
    with s2:
        scan_btn = st.button("🛡️ Run Security Audit", type="primary", use_container_width=True)

    if scan_btn:
        if not untrusted_prompt.strip():
            st.warning("Enter text to scan.")
        else:
            with st.spinner("Auditing prompt for adversarial patterns and sensitive exposure..."):
                report = scan_prompt_security(untrusted_prompt, use_ai=use_ai)
                store.log_activity(event_type="security_scan", security_risk=report["risk_level"])
                st.session_state["security_scan_report"] = report

    if "security_scan_report" in st.session_state:
        rep = st.session_state["security_scan_report"]
        risk = rep["risk_level"]
        accent = (
            PALETTE["error"]
            if risk == "High"
            else (PALETTE["warning"] if risk == "Medium" else PALETTE["success"])
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            render_kpi_card("Risk Classification", risk, f"Risk Index: {rep['risk_score']:.1f} / 100", accent)
        with r2:
            render_kpi_card(
                "Execution Gate",
                "Blocked" if not rep["is_safe_to_execute"] else "Allowed with Guardrails",
                "High-risk prompts require sanitization",
                accent,
            )
        with r3:
            render_kpi_card(
                "Threats Detected",
                str(len(rep["findings"])),
                "Rule-based + semantic checks",
                PALETTE["accent_primary"],
            )

        st.markdown("#### Identified Security Concerns")
        if not rep["findings"]:
            st.success("No known prompt injection, jailbreak, or credential exposure patterns were detected.")
        else:
            for item in rep["findings"]:
                st.markdown(
                    f"""
                    <div class="saas-card">
                        {risk_badge_html(item['severity'])} <strong>{item['category']}</strong><br/>
                        <span class="muted-text">Matched: <code>{item['pattern_matched']}</code></span>
                        <p style="margin:0.45rem 0 0 0;">{item['explanation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("#### Recommended Safer Delimiter-Isolated Alternative")
        st.text_area(
            "Safer Prompt",
            value=rep["safer_alternative"],
            height=170,
            label_visibility="collapsed",
        )
        st.info(rep["disclaimer"])


# ===========================================================================
# PAGE 6: SMART PROMPT LIBRARY & VERSION CONTROL
# ===========================================================================

def page_library() -> None:
    render_page_header(
        "Smart Prompt Library & Version Control",
        "Organize prompts into collections, inspect revision history, compare diffs, restore prior versions, and export.",
        badge_text="SQLite Versioned Vault",
    )

    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    with f1:
        search_q = st.text_input("Search prompts", placeholder="Search by title, keyword, or tag...")
    with f2:
        cat_filter = st.selectbox(
            "Category",
            [
                "All",
                "General",
                "Code & Software Engineering",
                "Data Science & ML",
                "Security & Compliance",
                "Content & Technical Writing",
                "Analysis & Strategy",
            ],
        )
    with f3:
        collections = store.list_collections()
        col_options = {"All Collections": 0}
        col_options.update({f"{c['name']} ({c['prompt_count']})": c["id"] for c in collections})
        sel_col = st.selectbox("Collection", list(col_options.keys()))
    with f4:
        export_fmt = st.selectbox("Export Format", ["json", "markdown", "csv"])
        exported_blob = store.export_prompts(fmt=export_fmt)
        st.download_button(
            "⬇️ Export",
            data=exported_blob,
            file_name=f"prompt_library.{export_fmt if export_fmt != 'markdown' else 'md'}",
            use_container_width=True,
        )

    with st.expander("➕ Create New Collection"):
        nc1, nc2, nc3 = st.columns([2, 3, 1])
        with nc1:
            new_col_name = st.text_input("Collection Name", placeholder="e.g., RAG Pipelines")
        with nc2:
            new_col_desc = st.text_input("Description", placeholder="Production retrieval & synthesis prompts")
        with nc3:
            st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
            if st.button("Create Folder", use_container_width=True):
                if new_col_name.strip():
                    try:
                        store.create_collection(new_col_name, new_col_desc)
                        st.success(f"Created collection '{new_col_name}'.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    prompts = store.list_prompts(
        search_query=search_q,
        category=cat_filter,
        collection_id=col_options[sel_col],
    )

    if not prompts:
        render_empty_state(
            "No Prompts Found in Library",
            "Optimize and auto-save a prompt in the Prompt Optimizer, or clear your current search filters.",
        )
        return

    for p in prompts:
        with st.expander(
            f"📌 {p.title}  |  v{p.current_version}  |  {p.category}  |  Score: {p.original_score:.1f} → {p.optimized_score:.1f}",
            expanded=False,
        ):
            meta_col, act_col = st.columns([3, 1])
            with meta_col:
                st.markdown(
                    f"**CNN:** `{p.cnn_prediction}` | **Bi-LSTM:** `{p.bilstm_prediction}` | "
                    f"**Strategy:** `{p.strategy_applied or 'N/A'}` | **Security:** {risk_badge_html(p.security_risk)}",
                    unsafe_allow_html=True,
                )
            with act_col:
                if st.button("🗑️ Delete", key=f"del_p_{p.id}", use_container_width=True):
                    store.delete_prompt(p.id)
                    st.rerun()

            p_left, p_right = st.columns(2)
            with p_left:
                st.markdown("**Original Prompt**")
                st.text_area(f"orig_{p.id}", value=p.original_prompt, height=160, disabled=True, label_visibility="collapsed")
            with p_right:
                st.markdown(f"**Current Optimized Prompt (v{p.current_version})**")
                edited_opt = st.text_area(f"opt_{p.id}", value=p.optimized_prompt, height=160, label_visibility="collapsed")
                rev_note = st.text_input("Revision summary", value="Manual refinement", key=f"note_{p.id}")
                if st.button("💾 Save as New Version", key=f"save_ver_{p.id}"):
                    store.create_new_version(
                        prompt_id=p.id,
                        new_prompt_text=edited_opt,
                        quality_score=p.optimized_score,
                        strategy_used="Manual Revision",
                        change_summary=rev_note,
                    )
                    st.success(f"Saved as v{p.current_version + 1}!")
                    st.rerun()

            versions = store.get_versions(p.id)
            st.markdown("##### Version History & Diff Compare")
            ver_df = pd.DataFrame(
                [
                    {
                        "Version": f"v{v.version_number}",
                        "Score": v.quality_score,
                        "Strategy": v.strategy_used,
                        "Summary": v.change_summary,
                        "Timestamp": v.created_at[:19].replace("T", " "),
                    }
                    for v in versions
                ]
            )
            st.dataframe(ver_df, use_container_width=True, hide_index=True)

            if len(versions) >= 2:
                v_labels = {f"v{v.version_number} ({v.change_summary})": v for v in versions}
                dc1, dc2, dc3 = st.columns([2, 2, 1])
                with dc1:
                    v_a_key = st.selectbox("Compare Version A", list(v_labels.keys()), index=1, key=f"va_{p.id}")
                with dc2:
                    v_b_key = st.selectbox("Compare Version B", list(v_labels.keys()), index=0, key=f"vb_{p.id}")
                with dc3:
                    st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
                    target_restore = v_labels[v_a_key].version_number
                    if st.button(f"↩️ Restore v{target_restore}", key=f"rest_{p.id}", use_container_width=True):
                        store.restore_version(p.id, target_restore)
                        st.success(f"Restored v{target_restore} as new latest version.")
                        st.rerun()

                diff_lines = list(
                    difflib.unified_diff(
                        v_labels[v_a_key].prompt_text.splitlines(),
                        v_labels[v_b_key].prompt_text.splitlines(),
                        fromfile=f"v{v_labels[v_a_key].version_number}",
                        tofile=f"v{v_labels[v_b_key].version_number}",
                        lineterm="",
                    )
                )
                st.code("\n".join(diff_lines) if diff_lines else "No text differences between selected versions.", language="diff")


# ===========================================================================
# PAGE 7: ANALYTICS & FEEDBACK INTELLIGENCE
# ===========================================================================

def page_analytics() -> None:
    render_page_header(
        "Platform Analytics & Feedback Intelligence",
        "Longitudinal quality trends, recurring prompt weaknesses, user satisfaction ratings, and training dataset export.",
        badge_text="Data-Driven Insights",
    )

    analytics = store.get_dashboard_analytics()
    fb = analytics["feedback_summary"]

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        render_kpi_card("Total Feedback Entries", str(fb["total"]), f"{fb['positive']} positive / {fb['negative']} negative")
    with a2:
        render_kpi_card("Average User Rating", f"{fb['avg_rating']:.2f} / 5.0", "Explicit user satisfaction", PALETTE["success"])
    with a3:
        render_kpi_card("Average Score Gain", f"{analytics['avg_improvement']:+.1f} pts", "Across all stored prompts", PALETTE["accent_secondary"])
    with a4:
        render_kpi_card("Security Detections", str(analytics["security_warnings"]), "Flagged across scans & runs", PALETTE["warning"])

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("#### Frequently Occurring Prompt Weaknesses")
        weaknesses = analytics.get("top_weaknesses", {})
        if not weaknesses:
            render_empty_state(
                "No Weakness Telemetry Yet",
                "Weakness frequency charts populate automatically as prompts are analyzed and rated.",
            )
        else:
            fig_w = go.Figure(
                go.Bar(
                    x=list(weaknesses.values()),
                    y=list(weaknesses.keys()),
                    orientation="h",
                    marker_color=PALETTE["accent_secondary"],
                )
            )
            apply_plotly_saas_theme(fig_w, height=340, title="Top Detected Structural Gaps")
            st.plotly_chart(fig_w, use_container_width=True)

    with ch2:
        st.markdown("#### User Rating Trend")
        history = fb.get("history", [])
        if not history:
            render_empty_state(
                "No User Feedback Submitted Yet",
                "Submit ratings in the Prompt Optimizer after running an optimization to track satisfaction trends.",
            )
        else:
            df_fb = pd.DataFrame(history)
            fig_fb = go.Figure(
                go.Scatter(
                    x=df_fb["index"],
                    y=df_fb["rating"],
                    mode="lines+markers",
                    line=dict(color=PALETTE["success"], width=3),
                    marker=dict(size=8),
                )
            )
            fig_fb.update_layout(yaxis=dict(range=[0.5, 5.5]))
            apply_plotly_saas_theme(fig_fb, height=340, title="Recent User Ratings (1–5 Scale)")
            st.plotly_chart(fig_fb, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Curated Feedback Dataset Export (For Future Model Fine-Tuning)")
    st.caption(
        "Export stored user feedback and human-rated prompt pairs in JSONL format for offline supervised fine-tuning or classifier retraining."
    )
    jsonl_data = store.export_feedback_jsonl()
    st.download_button(
        "⬇️ Download Feedback Training Dataset (.jsonl)",
        data=jsonl_data if jsonl_data else '{"info": "No feedback records stored yet"}\n',
        file_name="prompt_feedback_dataset.jsonl",
        mime="application/jsonl",
    )


# ===========================================================================
# Router
# ===========================================================================

PAGE_DISPATCH = {
    "Overview Dashboard": page_overview,
    "Prompt Optimizer": page_optimizer,
    "AI Playground": page_playground,
    "Deep Learning Benchmark": page_dl_benchmark,
    "Security Scanner": page_security_scanner,
    "Prompt Library": page_library,
    "Analytics": page_analytics,
}

PAGE_DISPATCH[selected_page]()