"""
playground/ab_testing.py
Member 3: AI Security & Evaluation Engineer
-------------------------------------------
Interactive Multi-Variant Prompt A/B Testing & Response Evaluation Engine.
Generates distinct prompt optimization variants, executes selected variants
against live LLM providers (Gemini first), scores real responses across
Relevance, Completeness, and Instruction-Following, and ranks the winner.
Never fabricates LLM outputs or evaluation metrics when API calls fail.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from playground.providers import get_provider

logger = logging.getLogger(__name__)


def generate_prompt_variants(base_prompt: str, num_variants: int = 3) -> List[Dict[str, str]]:
    """
    Generate multiple distinct prompt engineering variants for side-by-side testing
    using Gemini. Returns structured variants with strategy labels.
    """
    provider = get_provider("gemini")
    if not provider.is_configured():
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in .env. Configure it to generate prompt variants."
        )

    count = max(2, min(4, int(num_variants)))
    meta_prompt = f"""You are a Senior Prompt Architect. Given the BASE PROMPT below, generate {count} distinct, high-performing prompt variants using different prompt engineering strategies:
- Variant 1: CO-STAR Structured (Context, Objective, Style, Tone, Audience, Response Format)
- Variant 2: Step-by-Step Chain-of-Thought + Strict Constraints
- Variant 3: Role-Anchored + Few-Shot / Output Schema Specification
{f'- Variant 4: Concise High-Precision Executive Prompt' if count == 4 else ''}

Return ONLY valid JSON as a list of {count} objects, each with keys:
"name" (short label), "strategy" (strategy name), "prompt_text" (the full rewritten prompt).

BASE PROMPT:
\"\"\"
{base_prompt}
\"\"\"
"""
    completion = provider.generate(meta_prompt, temperature=0.5)
    if not completion.success:
        raise RuntimeError(f"Variant generation failed: {completion.error}")

    raw = re.sub(r"^```(?:json)?|```$", "", completion.output_text, flags=re.MULTILINE).strip()
    parsed = json.loads(raw)

    variants: List[Dict[str, str]] = [
        {
            "id": "baseline",
            "name": "Original Baseline",
            "strategy": "Unmodified Input",
            "prompt_text": base_prompt.strip(),
        }
    ]
    for idx, item in enumerate(parsed[:count], start=1):
        variants.append(
            {
                "id": f"variant_{idx}",
                "name": str(item.get("name", f"Variant {idx}")),
                "strategy": str(item.get("strategy", "Structured Optimization")),
                "prompt_text": str(item.get("prompt_text", "")).strip(),
            }
        )
    return variants


def _evaluate_response_with_llm(
    base_objective: str,
    prompt_used: str,
    model_response: str,
    provider_name: str = "gemini",
) -> Dict[str, Any]:
    """
    Evaluate an actual LLM response across Relevance, Completeness, and
    Instruction-Following (0–100 scale) using a structured judge prompt.
    """
    provider = get_provider(provider_name)
    judge_prompt = f"""You are an objective AI Response Evaluator. Score the MODEL RESPONSE below on a 0–100 scale across three criteria relative to the USER OBJECTIVE and PROMPT USED.

USER OBJECTIVE:
{base_objective[:1500]}

PROMPT USED:
{prompt_used[:2000]}

MODEL RESPONSE:
{model_response[:4000]}

Return ONLY valid JSON with exact keys:
{{
  "relevance": <float 0-100>,
  "completeness": <float 0-100>,
  "instruction_following": <float 0-100>,
  "rationale": "<one concise sentence explaining strengths or gaps>"
}}
"""
    judge_res = provider.generate(judge_prompt, temperature=0.1)
    if not judge_res.success:
        raise RuntimeError(f"Response evaluation failed: {judge_res.error}")

    raw = re.sub(r"^```(?:json)?|```$", "", judge_res.output_text, flags=re.MULTILINE).strip()
    scores = json.loads(raw)

    rel = round(max(0.0, min(100.0, float(scores.get("relevance", 0.0)))), 1)
    comp = round(max(0.0, min(100.0, float(scores.get("completeness", 0.0)))), 1)
    inst = round(max(0.0, min(100.0, float(scores.get("instruction_following", 0.0)))), 1)
    composite = round((rel + comp + inst) / 3.0, 1)

    return {
        "relevance": rel,
        "completeness": comp,
        "instruction_following": inst,
        "composite_score": composite,
        "rationale": str(scores.get("rationale", "")),
    }


def run_ab_experiment(
    base_prompt: str,
    variants: List[Dict[str, str]],
    provider_name: str = "gemini",
    model_name: str | None = None,
) -> Dict[str, Any]:
    """
    Public interface (Member 3 Contract):
    Execute 2 or more selected prompt variants against a real LLM provider,
    evaluate each live response, and identify the winning variant.
    """
    if len(variants) < 2:
        raise ValueError("Select at least 2 prompt variants to run an A/B comparison.")

    provider = get_provider(provider_name)
    if not provider.is_configured():
        raise RuntimeError(
            f"{provider.display_name} is not configured. Add the required API key in .env."
        )

    results: List[Dict[str, Any]] = []
    for idx, var in enumerate(variants):
        label = var.get("name") or f"Variant {idx + 1}"
        strategy = var.get("strategy") or "Custom"
        prompt_text = (var.get("prompt_text") or "").strip()
        if not prompt_text:
            continue

        completion = provider.generate(prompt_text, model=model_name, temperature=0.4)
        if not completion.success:
            results.append(
                {
                    "name": label,
                    "strategy": strategy,
                    "prompt_text": prompt_text,
                    "provider": completion.provider,
                    "model": completion.model,
                    "success": False,
                    "error": completion.error,
                    "output_text": "",
                    "latency_ms": completion.latency_ms,
                    "word_count": 0,
                    "metrics": {
                        "relevance": 0.0,
                        "completeness": 0.0,
                        "instruction_following": 0.0,
                        "composite_score": 0.0,
                        "rationale": f"Execution failed: {completion.error}",
                    },
                }
            )
            continue

        metrics = _evaluate_response_with_llm(
            base_objective=base_prompt,
            prompt_used=prompt_text,
            model_response=completion.output_text,
            provider_name=provider_name,
        )

        results.append(
            {
                "name": label,
                "strategy": strategy,
                "prompt_text": prompt_text,
                "provider": completion.provider,
                "model": completion.model,
                "success": True,
                "error": "",
                "output_text": completion.output_text,
                "latency_ms": completion.latency_ms,
                "word_count": completion.word_count,
                "metrics": metrics,
            }
        )

    successful = [r for r in results if r["success"]]
    if not successful:
        raise RuntimeError("All selected variants failed during live API execution.")

    winner = max(successful, key=lambda r: r["metrics"]["composite_score"])

    return {
        "base_prompt": base_prompt,
        "provider": provider.display_name,
        "winner": {
            "name": winner["name"],
            "strategy": winner["strategy"],
            "composite_score": winner["metrics"]["composite_score"],
            "rationale": winner["metrics"]["rationale"],
        },
        "results": results,
    }