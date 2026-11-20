"""
agents/state.py
Member 2: Agentic AI Engineer
-----------------------------
Shared LangGraph StateGraph TypedDict schema for the 7-agent autonomous
prompt optimization pipeline. Tracks inter-agent messages, Deep Learning
assessments, security checks, strategy selection, and bounded retry loops.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class PromptOptimizationState(TypedDict, total=False):
    # User inputs & configuration
    original_prompt: str
    target_model: str
    max_iterations: int

    # Agent 1: Prompt Analyzer outputs
    task_category: str
    user_objective: str
    detected_context: str
    missing_instructions: List[str]

    # Agent 2: Deep Learning Evaluation outputs (CNN + Bi-LSTM + Dimensions)
    original_score: float
    optimized_score: float
    cnn_prediction: Dict[str, Any]
    bilstm_prediction: Dict[str, Any]
    dimension_scores: Dict[str, float]
    weaknesses: List[str]
    recommendations: List[str]

    # Agent 3: Prompt Strategy outputs
    strategy_used: str
    strategy_instructions: str

    # Agent 4: Prompt Optimizer outputs
    candidate_prompt: str
    optimized_prompt: str
    iteration_count: int

    # Agent 5: Prompt Security outputs
    security_report: Dict[str, Any]
    security_blocked: bool

    # Agent 6: Quality Reviewer outputs
    review_passed: bool
    review_feedback: str

    # Agent 7: Final Response & Inter-Agent Traceability
    improvement_delta: float
    explanation: str
    agent_trace: List[Dict[str, Any]]
    error: str