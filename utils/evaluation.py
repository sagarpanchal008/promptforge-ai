"""
utils/evaluation.py
Member 1: Deep Learning Engineer
--------------------------------
Model evaluation metrics, confusion matrix loader, real Keras architecture
introspection (1D-CNN & Bi-LSTM), and multi-dimensional prompt quality
scoring (Clarity, Specificity, Context, Completeness) + missing instruction
detection. Never fabricates model predictions or training metrics.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Multi-Dimensional Prompt Quality & Missing Instruction Analyzer
# ---------------------------------------------------------------------------

def analyze_prompt_dimensions(prompt_text: str) -> Dict[str, Any]:
    """
    Evaluate a prompt across four interpretable engineering dimensions
    (0–100 scale) and detect missing structural instructions.
    """
    text = (prompt_text or "").strip()
    words = text.split()
    word_count = len(words)
    lower = text.lower()

    if word_count == 0:
        return {
            "dimension_scores": {
                "clarity": 0.0,
                "specificity": 0.0,
                "context": 0.0,
                "completeness": 0.0,
            },
            "missing_instructions": ["Non-empty task instruction"],
            "weaknesses": ["Empty prompt provided"],
            "recommendations": ["Provide a clear task objective, context, and desired output format."],
        }

    # 1. Clarity Score (action verbs, sentence structure, low ambiguity)
    action_verbs = (
        "analyze", "build", "create", "design", "explain", "generate", "implement",
        "list", "optimize", "refactor", "summarize", "write", "compare", "evaluate",
        "extract", "classify", "debug", "translate", "outline", "draft",
    )
    has_action_verb = any(re.search(rf"\b{v}\b", lower) for v in action_verbs)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_sentence_len = word_count / max(1, len(sentences))
    vague_terms = len(re.findall(r"\b(something|stuff|things|good|nice|etc|maybe|somehow)\b", lower))

    clarity = 45.0
    if has_action_verb:
        clarity += 25.0
    if 6 <= avg_sentence_len <= 28:
        clarity += 15.0
    if word_count >= 10:
        clarity += 10.0
    clarity -= min(25.0, vague_terms * 7.5)
    clarity = round(max(5.0, min(98.0, clarity)), 1)

    # 2. Specificity Score (constraints, numbers, technical terms, boundaries)
    has_numbers = bool(re.search(r"\b\d+\b", text))
    has_constraints = bool(
        re.search(r"\b(must|should|only|avoid|without|limit|at least|no more than|strictly|exact)\b", lower)
    )
    has_format = bool(
        re.search(r"\b(json|markdown|table|bullet|list|csv|code|python|sql|step-by-step|schema|sections)\b", lower)
    )
    specificity = 30.0
    if has_numbers:
        specificity += 18.0
    if has_constraints:
        specificity += 22.0
    if has_format:
        specificity += 20.0
    if word_count >= 25:
        specificity += 8.0
    specificity = round(max(5.0, min(98.0, specificity)), 1)

    # 3. Context Score (role/persona, audience, domain background, input data)
    has_role = bool(re.search(r"\b(act as|you are|as a|role:|senior|expert|architect|engineer|analyst)\b", lower))
    has_audience = bool(re.search(r"\b(for beginners|for executives|for developers|audience|user|client|students)\b", lower))
    has_background = bool(re.search(r"\b(context:|background:|given|using|based on|project|system|dataset)\b", lower))
    has_delimiters = any(d in text for d in ('"""', "```", "---", "###", "<", ">"))

    context = 28.0
    if has_role:
        context += 24.0
    if has_audience:
        context += 16.0
    if has_background:
        context += 18.0
    if has_delimiters:
        context += 12.0
    context = round(max(5.0, min(98.0, context)), 1)

    # 4. Completeness Score (examples/few-shot, edge cases, verification/output criteria)
    has_examples = bool(re.search(r"\b(example|e\.g\.|for instance|sample|input:|output:)\b", lower))
    has_edge_cases = bool(re.search(r"\b(edge case|error|fallback|if not|missing|invalid|null)\b", lower))
    has_criteria = bool(re.search(r"\b(criteria|ensure|verify|validate|tone|style|length)\b", lower))

    completeness = 32.0
    if has_format:
        completeness += 16.0
    if has_examples:
        completeness += 22.0
    if has_edge_cases:
        completeness += 14.0
    if has_criteria:
        completeness += 12.0
    completeness = round(max(5.0, min(98.0, completeness)), 1)

    # Missing instruction detection & actionable weaknesses
    missing_instructions: List[str] = []
    weaknesses: List[str] = []
    recommendations: List[str] = []

    if not has_role:
        missing_instructions.append("System Role / Persona definition")
        weaknesses.append("No explicit expert persona or domain role assigned")
        recommendations.append("Assign a clear domain role (e.g., 'Act as a Senior Data Engineer...').")

    if not has_format:
        missing_instructions.append("Output structure / format specification")
        weaknesses.append("Missing explicit output format (JSON, Markdown table, bullet list, code)")
        recommendations.append("Specify the exact output schema or structural layout expected.")

    if not has_constraints:
        missing_instructions.append("Scope boundaries & negative constraints")
        weaknesses.append("Lacks operational constraints or exclusion rules")
        recommendations.append("Add explicit constraints (length limits, rules, or things to avoid).")

    if not has_examples and word_count < 45:
        missing_instructions.append("Few-shot input/output example")
        weaknesses.append("No reference example provided to anchor response style")
        recommendations.append("Include at least one concise input/output example for consistent formatting.")

    if vague_terms > 0:
        weaknesses.append(f"Contains {vague_terms} ambiguous filler term(s)")
        recommendations.append("Replace vague words ('good', 'things', 'stuff') with measurable criteria.")

    if word_count < 12:
        weaknesses.append("Prompt is overly brief and under-specified")
        recommendations.append("Expand the prompt with background context, target audience, and success criteria.")

    if not weaknesses:
        weaknesses.append("No major structural weaknesses detected")
    if not recommendations:
        recommendations.append("Prompt is well-structured; consider adding edge-case handling instructions.")

    return {
        "dimension_scores": {
            "clarity": clarity,
            "specificity": specificity,
            "context": context,
            "completeness": completeness,
        },
        "missing_instructions": missing_instructions,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Deep Learning Benchmark & Saved Artifact Reader (No Fabricated Metrics)
# ---------------------------------------------------------------------------

def _locate_model_file(patterns: Tuple[str, ...]) -> Optional[Path]:
    if not MODELS_DIR.exists():
        return None
    for pat in patterns:
        matches = sorted(MODELS_DIR.glob(pat))
        if matches:
            return matches[0]
    return None


def inspect_keras_architecture(model_path: Optional[Path]) -> Dict[str, Any]:
    """Load a saved Keras model safely and extract real layer & parameter metadata."""
    if not model_path or not model_path.exists():
        return {"available": False, "path": None, "layers": [], "total_params": 0}

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(str(model_path), compile=False)
        layers_info = []
        for layer in model.layers:
            output_shape = getattr(layer, "output_shape", None)
            if output_shape is None and hasattr(layer, "output"):
                try:
                    output_shape = tuple(layer.output.shape)
                except Exception:
                    output_shape = "Dynamic"
            layers_info.append(
                {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "output_shape": str(output_shape),
                    "params": int(layer.count_params()),
                }
            )
        return {
            "available": True,
            "path": model_path.name,
            "layers": layers_info,
            "total_params": int(model.count_params()),
        }
    except Exception as exc:
        logger.warning("Could not inspect Keras model %s: %s", model_path, exc)
        return {"available": False, "path": model_path.name, "error": str(exc), "layers": [], "total_params": 0}


def load_saved_benchmark_results() -> Dict[str, Any]:
    """
    Read authentic evaluation artifacts from models/ (metrics JSON, training
    histories, confusion matrices, and Keras architectures). Never fabricates
    scores if artifacts have not been generated yet.
    """
    cnn_path = _locate_model_file(("*cnn*.keras", "*cnn*.h5", "*CNN*.keras", "*CNN*.h5"))
    bilstm_path = _locate_model_file(("*lstm*.keras", "*lstm*.h5", "*LSTM*.keras", "*LSTM*.h5"))

    cnn_arch = inspect_keras_architecture(cnn_path)
    bilstm_arch = inspect_keras_architecture(bilstm_path)

    # Search for any saved evaluation/history JSON files produced by notebooks 02 & 03
    metrics_data: Dict[str, Any] = {}
    if MODELS_DIR.exists():
        for json_file in sorted(MODELS_DIR.glob("*.json")):
            try:
                loaded = json.loads(json_file.read_text(encoding="utf-8"))
                metrics_data[json_file.stem] = loaded
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", json_file.name, exc)

    return {
        "models_dir_exists": MODELS_DIR.exists(),
        "cnn_architecture": cnn_arch,
        "bilstm_architecture": bilstm_arch,
        "saved_artifacts": metrics_data,
    }


def compute_classification_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Accuracy, Macro/Weighted Precision, Recall, F1, and Confusion Matrix."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = class_names or ["Poor", "Average", "Good"]
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_t, y_p, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_t, y_p, labels=list(range(len(labels)))).tolist()

    return {
        "accuracy": round(acc * 100.0, 2),
        "precision": round(float(prec) * 100.0, 2),
        "recall": round(float(rec) * 100.0, 2),
        "f1_score": round(float(f1) * 100.0, 2),
        "confusion_matrix": cm,
        "class_names": labels,
    }