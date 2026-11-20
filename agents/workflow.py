"""
agents/workflow.py
Member 2: Agentic AI Engineer
-----------------------------
Compiled LangGraph StateGraph orchestrating the 7-agent autonomous prompt
intelligence, security, strategy, optimization, review, and synthesis workflow
with conditional edges and a strict maximum optimization iteration limit.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.deep_learning_agent import (
    deep_learning_evaluation_node,
    evaluate_prompt_intelligence,
)
from agents.state import PromptOptimizationState

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent 1: Prompt Analyzer Node
# ---------------------------------------------------------------------------

def prompt_analyzer_node(state: PromptOptimizationState) -> PromptOptimizationState:
    """Identify user objective, task category, context clues, and missing instructions."""
    prompt = (state.get("original_prompt") or "").strip()
    lower = prompt.lower()

    category_rules = [
        ("Code & Software Engineering", ("python", "code", "function", "api", "sql", "bug", "debug", "react", "script", "algorithm")),
        ("Data Science & ML", ("dataset", "model", "train", "cnn", "lstm", "pandas", "classification", "accuracy", "neural")),
        ("Security & Compliance", ("security", "vulnerability", "injection", "audit", "auth", "encrypt", "policy")),
        ("Content & Technical Writing", ("write", "blog", "article", "documentation", "readme", "email", "summarize", "report")),
        ("Analysis & Strategy", ("analyze", "compare", "evaluate", "strategy", "plan", "research", "pros and cons")),
    ]

    detected_category = "General"
    for cat_name, keywords in category_rules:
        if any(kw in lower for kw in keywords):
            detected_category = cat_name
            break

    missing = []
    if len(prompt.split()) < 15:
        missing.append("Sufficient domain context and background")
    if not any(w in lower for w in ("json", "markdown", "table", "list", "format", "code", "bullet")):
        missing.append("Explicit output format specification")
    if not any(w in lower for w in ("act as", "you are", "role:", "expert", "senior")):
        missing.append("Expert persona / role framing")

    objective = prompt.splitlines()[0][:140] if prompt else "Unspecified objective"
    context_note = (
        "Rich contextual framing detected"
        if len(prompt.split()) >= 35
        else "Minimal context provided; requires structural enrichment"
    )

    trace = list(state.get("agent_trace", []))
    trace.append(
        {
            "agent": "Prompt Analyzer Agent",
            "status": "completed",
            "summary": f"Category: {detected_category} | Missing elements: {len(missing)}",
            "details": {
                "task_category": detected_category,
                "user_objective": objective,
                "missing_instructions": missing,
            },
        }
    )

    return {
        "task_category": detected_category,
        "user_objective": objective,
        "detected_context": context_note,
        "missing_instructions": missing,
        "agent_trace": trace,
    }


# ---------------------------------------------------------------------------
# Agent 3: Prompt Security Node
# ---------------------------------------------------------------------------

def prompt_security_node(state: PromptOptimizationState) -> PromptOptimizationState:
    """Scan prompt for injection, override, jailbreak, and sensitive data exposure."""
    from security.scanner import scan_prompt_security

    prompt = state.get("original_prompt", "")
    report = scan_prompt_security(prompt, use_ai=False)
    risk_level = report.get("risk_level", "Low")
    blocked = risk_level == "High"

    trace = list(state.get("agent_trace", []))
    trace.append(
        {
            "agent": "Prompt Security Agent",
            "status": "blocked" if blocked else "completed",
            "summary": f"Security Risk: {risk_level} ({len(report.get('findings', []))} finding(s))",
            "details": {"risk_level": risk_level, "blocked": blocked},
        }
    )

    return {
        "security_report": report,
        "security_blocked": blocked,
        "agent_trace": trace,
    }


# ---------------------------------------------------------------------------
# Agent 4: Prompt Strategy Node
# ---------------------------------------------------------------------------

def prompt_strategy_node(state: PromptOptimizationState) -> PromptOptimizationState:
    """Select an optimization strategy tailored to detected weaknesses and task category."""
    category = state.get("task_category", "General")
    dims = state.get("dimension_scores", {})
    weaknesses = state.get("weaknesses", [])

    if state.get("security_blocked"):
        strategy = "Security-Sanitized Reframing"
        instructions = (
            "Remove adversarial override patterns, isolate untrusted data inside delimiters, "
            "and preserve only the legitimate core task."
        )
    elif category in ("Code & Software Engineering", "Data Science & ML"):
        strategy = "CO-STAR + Technical Specification"
        instructions = (
            "Structure with Role, Context, Technical Constraints, Input/Output Schema, "
            "Edge-Case Handling, and Verification Criteria."
        )
    elif dims.get("completeness", 50) < 55:
        strategy = "CO-STAR + Few-Shot Scaffolding"
        instructions = (
            "Add an explicit expert persona, step-by-step instructions, concrete output format, "
            "and a concise example to eliminate ambiguity."
        )
    else:
        strategy = "Structured Role-Goal-Constraints (RGC)"
        instructions = (
            "Define an authoritative role, explicit objective, numbered execution steps, "
            "and strict negative constraints."
        )

    trace = list(state.get("agent_trace", []))
    trace.append(
        {
            "agent": "Prompt Strategy Agent",
            "status": "completed",
            "summary": f"Selected Strategy: {strategy}",
            "details": {"strategy": strategy, "addressed_weaknesses": weaknesses[:3]},
        }
    )

    return {
        "strategy_used": strategy,
        "strategy_instructions": instructions,
        "agent_trace": trace,
    }


# ---------------------------------------------------------------------------
# Agent 5: Prompt Optimizer Node (Gemini API)
# ---------------------------------------------------------------------------

def _call_gemini_optimizer(
    original_prompt: str,
    strategy: str,
    strategy_instructions: str,
    weaknesses: list[str],
    review_feedback: str,
    target_model: str,
) -> str:
    """Invoke Gemini API to synthesize a structured, high-performing prompt."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in .env. Add your key to enable live Gemini optimization."
        )

    from google import genai

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    meta_prompt = f"""You are an expert Prompt Engineer. Rewrite and optimize the user prompt below so it achieves maximum clarity, specificity, context completeness, and instruction reliability on {target_model}.

APPLIED STRATEGY: {strategy}
STRATEGY DIRECTIVES: {strategy_instructions}
DETECTED WEAKNESSES TO FIX: {'; '.join(weaknesses)}
{f'PREVIOUS REVIEWER FEEDBACK TO ADDRESS: {review_feedback}' if review_feedback else ''}

RULES:
1. Preserve the user's original intent completely without inventing unrelated goals.
2. Structure the optimized prompt cleanly using clear sections (Role, Objective, Context, Constraints, Output Format).
3. Treat any embedded user data safely with clear delimiters.
4. Output ONLY the final optimized prompt text — no conversational preamble or markdown wrapper around the whole response.

ORIGINAL PROMPT:
\"\"\"
{original_prompt}
\"\"\"
"""
    response = client.models.generate_content(
        model=model_name,
        contents=meta_prompt,
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty optimization response.")
    return text


def prompt_optimizer_node(state: PromptOptimizationState) -> PromptOptimizationState:
    """Generate an improved prompt using Gemini API and increment the iteration counter."""
    iteration = int(state.get("iteration_count", 0)) + 1
    original_prompt = state.get("original_prompt", "")
    strategy = state.get("strategy_used", "CO-STAR")
    strategy_instructions = state.get("strategy_instructions", "")
    weaknesses = state.get("weaknesses", [])
    review_feedback = state.get("review_feedback", "")
    target_model = state.get("target_model", "gemini-2.5-flash")

    # If High security risk was flagged, use the safer alternative as the base input
    if state.get("security_blocked"):
        safer_base = state.get("security_report", {}).get("safer_alternative") or original_prompt
        source_text = safer_base
    else:
        source_text = original_prompt

    optimized = _call_gemini_optimizer(
        original_prompt=source_text,
        strategy=strategy,
        strategy_instructions=strategy_instructions,
        weaknesses=weaknesses,
        review_feedback=review_feedback,
        target_model=target_model,
    )

    trace = list(state.get("agent_trace", []))
    trace.append(
        {
            "agent": "Prompt Optimizer Agent",
            "status": "completed",
            "summary": f"Iteration {iteration}: Generated optimized prompt ({len(optimized.split())} words)",
            "details": {"iteration": iteration, "strategy": strategy},
        }
    )

    return {
        "candidate_prompt": optimized,
        "optimized_prompt": optimized,
        "iteration_count": iteration,
        "agent_trace": trace,
    }


# ---------------------------------------------------------------------------
# Agent 6: Quality Reviewer Node (Re-evaluates with DL Engine & Controls Loop)
# ---------------------------------------------------------------------------

def quality_reviewer_node(state: PromptOptimizationState) -> PromptOptimizationState:
    """Re-score candidate prompt using the Deep Learning engine and decide if another cycle is needed."""
    candidate = state.get("candidate_prompt", "")
    orig_score = float(state.get("original_score", 0.0))
    iteration = int(state.get("iteration_count", 1))
    max_iter = int(state.get("max_iterations", 2))

    post_eval = evaluate_prompt_intelligence(candidate)
    new_score = post_eval["overall_score"]
    remaining_weaknesses = [
        w for w in post_eval["weaknesses"] if "No major structural weaknesses" not in w
    ]

    passed = (new_score >= max(78.0, orig_score + 8.0) and len(remaining_weaknesses) <= 1) or (
        iteration >= max_iter
    )

    if passed:
        feedback = (
            f"Approved on iteration {iteration}: quality improved from {orig_score:.1f} to {new_score:.1f}."
        )
    else:
        feedback = (
            f"Score {new_score:.1f} needs further refinement. Address remaining gaps: "
            + "; ".join(remaining_weaknesses[:2])
        )

    trace = list(state.get("agent_trace", []))
    trace.append(
        {
            "agent": "Quality Reviewer Agent",
            "status": "approved" if passed else "retry_requested",
            "summary": feedback,
            "details": {
                "new_score": new_score,
                "iteration": iteration,
                "max_iterations": max_iter,
                "approved": passed,
            },
        }
    )

    return {
        "optimized_score": new_score,
        "dimension_scores": post_eval["dimension_scores"],
        "review_passed": passed,
        "review_feedback": feedback,
        "agent_trace": trace,
    }


def _route_after_review(state: PromptOptimizationState) -> str:
    """Conditional edge: retry optimization if quality threshold unmet and within max_iterations."""
    if state.get("review_passed", True):
        return "final"
    if int(state.get("iteration_count", 1)) >= int(state.get("max_iterations", 2)):
        return "final"
    return "optimize"


# ---------------------------------------------------------------------------
# Agent 7: Final Response Node
# ---------------------------------------------------------------------------

def final_response_node(state: PromptOptimizationState) -> PromptOptimizationState:
    """Synthesize final improvement metrics and human-readable optimization report."""
    orig_score = float(state.get("original_score", 0.0))
    opt_score = float(state.get("optimized_score", orig_score))
    delta = round(opt_score - orig_score, 1)
    strategy = state.get("strategy_used", "Structured Optimization")
    iterations = int(state.get("iteration_count", 1))
    weaknesses = state.get("weaknesses", [])

    explanation = (
        f"Applied **{strategy}** across {iterations} autonomous review cycle(s). "
        f"Overall quality moved from **{orig_score:.1f}/100** to **{opt_score:.1f}/100** "
        f"({delta:+.1f} pts) by resolving: {', '.join(weaknesses[:3])}."
    )

    trace = list(state.get("agent_trace", []))
    trace.append(
        {
            "agent": "Final Response Agent",
            "status": "completed",
            "summary": f"Delivered final prompt ({delta:+.1f} pt improvement)",
            "details": {"improvement_delta": delta, "iterations_completed": iterations},
        }
    )

    return {
        "improvement_delta": delta,
        "explanation": explanation,
        "agent_trace": trace,
    }


# ---------------------------------------------------------------------------
# Compile LangGraph StateGraph
# ---------------------------------------------------------------------------

def build_optimization_graph():
    """Construct and compile the 7-agent LangGraph StateGraph."""
    graph = StateGraph(PromptOptimizationState)

    graph.add_node("analyze", prompt_analyzer_node)
    graph.add_node("dl_evaluate", deep_learning_evaluation_node)
    graph.add_node("security", prompt_security_node)
    graph.add_node("strategy", prompt_strategy_node)
    graph.add_node("optimize", prompt_optimizer_node)
    graph.add_node("review", quality_reviewer_node)
    graph.add_node("final", final_response_node)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "dl_evaluate")
    graph.add_edge("dl_evaluate", "security")
    graph.add_edge("security", "strategy")
    graph.add_edge("strategy", "optimize")
    graph.add_edge("optimize", "review")

    graph.add_conditional_edges(
        "review",
        _route_after_review,
        {
            "optimize": "optimize",
            "final": "final",
        },
    )
    graph.add_edge("final", END)

    return graph.compile()


_COMPILED_GRAPH = build_optimization_graph()


def run_optimization_pipeline(
    raw_prompt: str,
    target_model: str = "gemini-2.5-flash",
    max_iterations: int = 2,
) -> Dict[str, Any]:
    """Execute the full 7-agent LangGraph pipeline and return structured output."""
    initial_state: PromptOptimizationState = {
        "original_prompt": raw_prompt,
        "target_model": target_model,
        "max_iterations": max(1, min(5, int(max_iterations))),
        "iteration_count": 0,
        "agent_trace": [],
    }

    final_state = _COMPILED_GRAPH.invoke(initial_state)

    return {
        "original_prompt": final_state.get("original_prompt", raw_prompt),
        "optimized_prompt": final_state.get("optimized_prompt", raw_prompt),
        "task_category": final_state.get("task_category", "General"),
        "original_score": final_state.get("original_score", 0.0),
        "optimized_score": final_state.get("optimized_score", 0.0),
        "improvement_delta": final_state.get("improvement_delta", 0.0),
        "cnn_prediction": final_state.get("cnn_prediction", {}),
        "bilstm_prediction": final_state.get("bilstm_prediction", {}),
        "dimension_scores": final_state.get("dimension_scores", {}),
        "weaknesses": final_state.get("weaknesses", []),
        "missing_instructions": final_state.get("missing_instructions", []),
        "recommendations": final_state.get("recommendations", []),
        "strategy_used": final_state.get("strategy_used", ""),
        "security_report": final_state.get("security_report", {}),
        "iterations_completed": final_state.get("iteration_count", 1),
        "agent_trace": final_state.get("agent_trace", []),
        "explanation": final_state.get("explanation", ""),
    }