"""
backend/schemas.py
Member 4: Backend & Database Engineer
-------------------------------------
Pydantic v2 request and response validation models for FastAPI endpoints.
Ensures strict input bounds, clean error reporting, and backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Optimization & Prompt Intelligence Schemas
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=12000, description="Raw user prompt to analyze and optimize")
    target_model: str = Field(default="gemini-2.5-flash", max_length=64)
    max_iterations: int = Field(default=2, ge=1, le=5)
    save_to_library: bool = Field(default=True)
    collection_id: Optional[int] = Field(default=1)
    title: Optional[str] = Field(default=None, max_length=120)


class OptimizeResponse(BaseModel):
    original_prompt: str
    optimized_prompt: str
    task_category: str
    original_score: float
    optimized_score: float
    improvement_delta: float
    cnn_prediction: Dict[str, Any]
    bilstm_prediction: Dict[str, Any]
    dimension_scores: Dict[str, float]
    weaknesses: List[str]
    missing_instructions: List[str]
    recommendations: List[str]
    strategy_used: str
    security_report: Dict[str, Any]
    iterations_completed: int
    agent_trace: List[Dict[str, Any]]
    explanation: str
    saved_prompt_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Security Scanner Schemas
# ---------------------------------------------------------------------------

class SecurityScanRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=15000)
    use_ai_analysis: bool = Field(default=True)


class SecurityFinding(BaseModel):
    category: str
    severity: Literal["Low", "Medium", "High"]
    pattern_matched: str
    explanation: str


class SecurityScanResponse(BaseModel):
    risk_level: Literal["Low", "Medium", "High"]
    risk_score: float
    is_safe_to_execute: bool
    findings: List[SecurityFinding]
    safer_alternative: str
    disclaimer: str


# ---------------------------------------------------------------------------
# A/B Playground Schemas
# ---------------------------------------------------------------------------

class GenerateVariantsRequest(BaseModel):
    base_prompt: str = Field(..., min_length=3, max_length=8000)
    num_variants: int = Field(default=3, ge=2, le=4)


class ABCompareRequest(BaseModel):
    base_prompt: str = Field(..., min_length=3, max_length=8000)
    variants: List[Dict[str, str]] = Field(..., min_length=2, max_length=4)
    provider: str = Field(default="gemini")


# ---------------------------------------------------------------------------
# Prompt Library, Version Control & Collections Schemas
# ---------------------------------------------------------------------------

class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(default="", max_length=250)
    color: str = Field(default="#2788F5", pattern=r"^#[0-9A-Fa-f]{6}$")


class PromptCreateRequest(BaseModel):
    title: str = Field(default="Saved Prompt", max_length=120)
    original_prompt: str = Field(..., min_length=1, max_length=12000)
    optimized_prompt: str = Field(..., min_length=1, max_length=12000)
    category: str = Field(default="General", max_length=64)
    target_model: str = Field(default="gemini-2.5-flash", max_length=64)
    collection_id: Optional[int] = Field(default=1)
    original_score: float = Field(default=0.0, ge=0.0, le=100.0)
    optimized_score: float = Field(default=0.0, ge=0.0, le=100.0)
    cnn_prediction: str = Field(default="Unknown")
    bilstm_prediction: str = Field(default="Unknown")
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    weaknesses: List[str] = Field(default_factory=list)
    strategy_applied: str = Field(default="")
    security_risk: str = Field(default="Low")
    tags: List[str] = Field(default_factory=list)


class VersionCreateRequest(BaseModel):
    prompt_text: str = Field(..., min_length=1, max_length=12000)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    strategy_used: str = Field(default="Manual Edit")
    change_summary: str = Field(default="Manual revision", max_length=200)


# ---------------------------------------------------------------------------
# Feedback Schemas
# ---------------------------------------------------------------------------

class FeedbackCreateRequest(BaseModel):
    prompt_id: Optional[int] = None
    original_prompt: str = Field(default="")
    optimized_prompt: str = Field(default="")
    thumbs_up: bool = True
    rating: int = Field(default=5, ge=1, le=5)
    comment: str = Field(default="", max_length=1000)
    detected_weaknesses: List[str] = Field(default_factory=list)
    strategy_used: str = Field(default="")