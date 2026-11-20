"""
backend/api.py
Member 4: Backend & Database Engineer
-------------------------------------
FastAPI REST server for Intelligent Prompt Optimization Assistant 2.0.
Preserves backward-compatible v1.0 endpoints while exposing v2.0 multi-agent
optimization, hybrid security scanning, A/B playground execution, versioned
prompt library CRUD, feedback capture, and live dashboard analytics.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from backend.schemas import (
    ABCompareRequest,
    CollectionCreateRequest,
    FeedbackCreateRequest,
    GenerateVariantsRequest,
    OptimizeRequest,
    PromptCreateRequest,
    SecurityScanRequest,
    VersionCreateRequest,
)
from database.models import FeedbackRecord, PromptRecord
from database.prompt_store import PromptStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("prompt_optimizer.api")

app = FastAPI(
    title="Intelligent Prompt Optimization Assistant 2.0 API",
    description="AI-Powered Prompt Intelligence, Evaluation, Security & Optimization Platform",
    version="2.0.0",
)

# Restrict CORS to local development origins by default
allowed_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

store = PromptStore()


# ------------------------------------------------------------------
# Health & System Status
# ------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "version": "2.0.0",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "database_path": str(store.db_path),
    }


# ------------------------------------------------------------------
# Multi-Agent Optimization Pipeline
# ------------------------------------------------------------------

@app.post("/optimize")
def optimize_prompt(req: OptimizeRequest) -> dict:
    try:
        from agents.workflow import run_optimization_pipeline

        result = run_optimization_pipeline(
            raw_prompt=req.prompt,
            target_model=req.target_model,
            max_iterations=req.max_iterations,
        )

        saved_id: Optional[int] = None
        if req.save_to_library:
            rec = PromptRecord(
                title=req.title or "Untitled Prompt",
                original_prompt=result["original_prompt"],
                optimized_prompt=result["optimized_prompt"],
                category=result.get("task_category", "General"),
                target_model=req.target_model,
                collection_id=req.collection_id or 1,
                original_score=result.get("original_score", 0.0),
                optimized_score=result.get("optimized_score", 0.0),
                cnn_prediction=result.get("cnn_prediction", {}).get("label", "Unknown"),
                bilstm_prediction=result.get("bilstm_prediction", {}).get("label", "Unknown"),
                dimension_scores=result.get("dimension_scores", {}),
                weaknesses=result.get("weaknesses", []),
                strategy_applied=result.get("strategy_used", ""),
                security_risk=result.get("security_report", {}).get("risk_level", "Low"),
                optimization_settings={
                    "target_model": req.target_model,
                    "max_iterations": req.max_iterations,
                },
            )
            saved = store.save_prompt(rec, change_summary="Auto-saved from Multi-Agent Optimizer")
            saved_id = saved.id

        store.log_activity(
            event_type="optimize",
            category=result.get("task_category", "General"),
            original_score=result.get("original_score", 0.0),
            optimized_score=result.get("optimized_score", 0.0),
            security_risk=result.get("security_report", {}).get("risk_level", "Low"),
        )

        result["saved_prompt_id"] = saved_id
        return result
    except Exception as exc:
        logger.exception("Optimization pipeline failed")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc


# ------------------------------------------------------------------
# Security Scanner Endpoints
# ------------------------------------------------------------------

@app.post("/security/scan")
def scan_prompt(req: SecurityScanRequest) -> dict:
    try:
        from security.scanner import scan_prompt_security

        report = scan_prompt_security(req.prompt, use_ai=req.use_ai_analysis)
        store.log_activity(
            event_type="security_scan",
            security_risk=report.get("risk_level", "Low"),
        )
        return report
    except Exception as exc:
        logger.exception("Security scan failed")
        raise HTTPException(status_code=500, detail=f"Security scan failed: {exc}") from exc


# ------------------------------------------------------------------
# AI Playground & A/B Testing Endpoints
# ------------------------------------------------------------------

@app.post("/playground/variants")
def generate_variants(req: GenerateVariantsRequest) -> dict:
    try:
        from playground.ab_testing import generate_prompt_variants

        variants = generate_prompt_variants(req.base_prompt, num_variants=req.num_variants)
        return {"base_prompt": req.base_prompt, "variants": variants}
    except Exception as exc:
        logger.exception("Variant generation failed")
        raise HTTPException(status_code=500, detail=f"Variant generation failed: {exc}") from exc


@app.post("/playground/compare")
def compare_variants(req: ABCompareRequest) -> dict:
    try:
        from playground.ab_testing import run_ab_experiment

        comparison = run_ab_experiment(
            base_prompt=req.base_prompt,
            variants=req.variants,
            provider_name=req.provider,
        )
        store.log_activity(event_type="ab_test")
        return comparison
    except Exception as exc:
        logger.exception("A/B comparison failed")
        raise HTTPException(status_code=500, detail=f"A/B comparison failed: {exc}") from exc


# ------------------------------------------------------------------
# Prompt Library, Collections & Version Control
# ------------------------------------------------------------------

@app.get("/collections")
def get_collections() -> list:
    return store.list_collections()


@app.post("/collections")
def create_collection(req: CollectionCreateRequest) -> dict:
    try:
        col = store.create_collection(req.name, req.description, req.color)
        return col.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/prompts")
def list_prompts(
    q: str = Query(default=""),
    category: Optional[str] = Query(default=None),
    collection_id: Optional[int] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list:
    items = store.list_prompts(
        search_query=q, category=category, collection_id=collection_id, limit=limit
    )
    return [item.to_dict() for item in items]


@app.post("/prompts")
def create_prompt(req: PromptCreateRequest) -> dict:
    record = PromptRecord(
        title=req.title,
        original_prompt=req.original_prompt,
        optimized_prompt=req.optimized_prompt,
        category=req.category,
        target_model=req.target_model,
        collection_id=req.collection_id,
        original_score=req.original_score,
        optimized_score=req.optimized_score,
        cnn_prediction=req.cnn_prediction,
        bilstm_prediction=req.bilstm_prediction,
        dimension_scores=req.dimension_scores,
        weaknesses=req.weaknesses,
        strategy_applied=req.strategy_applied,
        security_risk=req.security_risk,
        tags=req.tags,
    )
    saved = store.save_prompt(record)
    return saved.to_dict()


@app.get("/prompts/{prompt_id}")
def get_prompt(prompt_id: int) -> dict:
    item = store.get_prompt(prompt_id)
    if not item:
        raise HTTPException(status_code=404, detail="Prompt not found.")
    return item.to_dict()


@app.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: int) -> dict:
    deleted = store.delete_prompt(prompt_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prompt not found.")
    return {"deleted": True, "id": prompt_id}


@app.get("/prompts/{prompt_id}/versions")
def get_prompt_versions(prompt_id: int) -> list:
    versions = store.get_versions(prompt_id)
    return [v.to_dict() for v in versions]


@app.post("/prompts/{prompt_id}/versions")
def add_prompt_version(prompt_id: int, req: VersionCreateRequest) -> dict:
    try:
        ver = store.create_new_version(
            prompt_id=prompt_id,
            new_prompt_text=req.prompt_text,
            quality_score=req.quality_score,
            strategy_used=req.strategy_used,
            change_summary=req.change_summary,
        )
        return ver.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/prompts/{prompt_id}/restore/{version_number}")
def restore_prompt_version(prompt_id: int, version_number: int) -> dict:
    try:
        restored = store.restore_version(prompt_id, version_number)
        return restored.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/export/prompts", response_class=PlainTextResponse)
def export_prompts(fmt: str = Query(default="json", pattern="^(json|markdown|csv)$")) -> str:
    return store.export_prompts(fmt=fmt)


# ------------------------------------------------------------------
# Feedback & Analytics
# ------------------------------------------------------------------

@app.post("/feedback")
def submit_feedback(req: FeedbackCreateRequest) -> dict:
    fb = FeedbackRecord(
        prompt_id=req.prompt_id,
        original_prompt=req.original_prompt,
        optimized_prompt=req.optimized_prompt,
        thumbs_up=req.thumbs_up,
        rating=req.rating,
        comment=req.comment,
        detected_weaknesses=req.detected_weaknesses,
        strategy_used=req.strategy_used,
    )
    saved = store.save_feedback(fb)
    return saved.to_dict()


@app.get("/analytics")
def get_analytics() -> dict:
    return store.get_dashboard_analytics()