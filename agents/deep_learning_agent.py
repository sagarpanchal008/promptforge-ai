"""
agents/deep_learning_agent.py
Members 1 & 2: Deep Learning Engineer & Agentic AI Engineer
-----------------------------------------------------------
Loads the trained 1D-CNN and Bi-LSTM Keras models alongside the serialized
tokenizer (models/tokenizer.pkl), computes real class probability vectors
(Poor, Average, Good), combines them with multi-dimensional quality metrics
(Clarity, Specificity, Context, Completeness), and updates LangGraph state.
Never fabricates neural predictions if model files are absent.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from agents.state import PromptOptimizationState
from utils.evaluation import analyze_prompt_dimensions

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CLASS_LABELS = ["Poor", "Average", "Good"]
MAX_SEQ_LEN = 120

_MODEL_CACHE: Dict[str, Any] = {
    "loaded": False,
    "cnn": None,
    "bilstm": None,
    "tokenizer": None,
    "cnn_name": None,
    "bilstm_name": None,
}


def _find_first(patterns: Tuple[str, ...]) -> Optional[Path]:
    if not MODELS_DIR.exists():
        return None
    for pat in patterns:
        matches = sorted(MODELS_DIR.glob(pat))
        if matches:
            return matches[0]
    return None


def _register_custom_unpickle_classes() -> None:
    """Attach notebook-pickled tokenizer classes from utils.preprocessing onto __main__."""
    try:
        import utils.preprocessing as prep

        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            for attr_name in dir(prep):
                obj = getattr(prep, attr_name)
                if isinstance(obj, type) and not hasattr(main_mod, attr_name):
                    setattr(main_mod, attr_name, obj)
    except Exception as exc:
        logger.debug("Custom unpickle registration skipped: %s", exc)


def _load_artifacts() -> None:
    """Lazy-load and cache saved CNN, Bi-LSTM, and tokenizer artifacts."""
    if _MODEL_CACHE["loaded"]:
        return

    _MODEL_CACHE["loaded"] = True
    tok_path = _find_first(("*tokenizer*.pkl", "*.pkl"))
    cnn_path = _find_first(("*cnn*.keras", "*cnn*.h5", "*CNN*.keras", "*CNN*.h5"))
    bilstm_path = _find_first(("*lstm*.keras", "*lstm*.h5", "*LSTM*.keras", "*LSTM*.h5"))

    if tok_path and tok_path.exists():
        try:
            _register_custom_unpickle_classes()
            with open(tok_path, "rb") as f:
                _MODEL_CACHE["tokenizer"] = pickle.load(f)
            logger.info("Loaded tokenizer from %s", tok_path.name)
        except Exception as exc:
            logger.warning("Failed to load tokenizer %s: %s", tok_path, exc)

    try:
        import tensorflow as tf

        if cnn_path and cnn_path.exists():
            _MODEL_CACHE["cnn"] = tf.keras.models.load_model(str(cnn_path), compile=False)
            _MODEL_CACHE["cnn_name"] = cnn_path.name
            logger.info("Loaded 1D-CNN model from %s", cnn_path.name)

        if bilstm_path and bilstm_path.exists():
            _MODEL_CACHE["bilstm"] = tf.keras.models.load_model(str(bilstm_path), compile=False)
            _MODEL_CACHE["bilstm_name"] = bilstm_path.name
            logger.info("Loaded Bi-LSTM model from %s", bilstm_path.name)
    except Exception as exc:
        logger.warning("TensorFlow model loading skipped/failed: %s", exc)


def _vectorize_prompt(text: str) -> Optional[np.ndarray]:
    """Convert raw text into a padded (1, 120) integer sequence using the saved tokenizer."""
    tok = _MODEL_CACHE.get("tokenizer")
    if tok is None:
        return None

    try:
        if hasattr(tok, "texts_to_sequences"):
            seqs = tok.texts_to_sequences([text])
        elif hasattr(tok, "transform"):
            arr = np.asarray(tok.transform([text]), dtype="int32")
            return arr[:, :MAX_SEQ_LEN]
        elif hasattr(tok, "encode"):
            seqs = [tok.encode(text)]
        elif isinstance(tok, dict) and "word_index" in tok:
            word_idx = tok["word_index"]
            seqs = [[word_idx.get(w.lower(), 1) for w in text.split()]]
        else:
            return None

        seq = seqs[0] if seqs else []
        padded = np.zeros((1, MAX_SEQ_LEN), dtype="int32")
        truncated = seq[:MAX_SEQ_LEN]
        if truncated:
            padded[0, : len(truncated)] = truncated
        return padded
    except Exception as exc:
        logger.warning("Prompt vectorization failed: %s", exc)
        return None


def _predict_single_model(model: Any, padded_seq: Optional[np.ndarray], model_tag: str) -> Dict[str, Any]:
    """Run real forward-pass inference on a Keras model or report unavailable status."""
    if model is None or padded_seq is None:
        return {
            "available": False,
            "model": model_tag,
            "label": "Unavailable",
            "confidence": 0.0,
            "probabilities": {},
            "score": None,
        }

    try:
        probs = model.predict(padded_seq, verbose=0)[0]
        probs = np.asarray(probs, dtype=float)
        if probs.ndim == 0 or len(probs) < 3:
            val = float(probs[0]) if probs.ndim > 0 else float(probs)
            label = "Good" if val >= 0.66 else ("Average" if val >= 0.33 else "Poor")
            return {
                "available": True,
                "model": model_tag,
                "label": label,
                "confidence": round(val * 100.0, 2),
                "probabilities": {"raw": round(val, 4)},
                "score": round(val * 100.0, 2),
            }

        idx = int(np.argmax(probs[:3]))
        label = CLASS_LABELS[idx]
        expected_score = float(probs[0] * 25.0 + probs[1] * 62.0 + probs[2] * 95.0)
        return {
            "available": True,
            "model": model_tag,
            "label": label,
            "confidence": round(float(probs[idx]) * 100.0, 2),
            "probabilities": {
                "Poor": round(float(probs[0]) * 100.0, 2),
                "Average": round(float(probs[1]) * 100.0, 2),
                "Good": round(float(probs[2]) * 100.0, 2),
            },
            "score": round(expected_score, 2),
        }
    except Exception as exc:
        logger.warning("%s inference error: %s", model_tag, exc)
        return {
            "available": False,
            "model": model_tag,
            "label": "Error",
            "confidence": 0.0,
            "probabilities": {},
            "score": None,
        }


def evaluate_prompt_intelligence(prompt_text: str) -> Dict[str, Any]:
    """
    Public interface (Member 1 Contract):
    Run 1D-CNN + Bi-LSTM inference + 4-dimension quality breakdown on any prompt.
    """
    _load_artifacts()
    padded = _vectorize_prompt(prompt_text)

    cnn_res = _predict_single_model(
        _MODEL_CACHE.get("cnn"), padded, _MODEL_CACHE.get("cnn_name") or "1D-CNN"
    )
    bilstm_res = _predict_single_model(
        _MODEL_CACHE.get("bilstm"), padded, _MODEL_CACHE.get("bilstm_name") or "Bi-LSTM"
    )
    dim_report = analyze_prompt_dimensions(prompt_text)
    dims = dim_report["dimension_scores"]
    dim_avg = sum(dims.values()) / max(1, len(dims))

    model_scores = [s for s in (cnn_res["score"], bilstm_res["score"]) if s is not None]
    if model_scores:
        dl_avg = sum(model_scores) / len(model_scores)
        overall_score = round(0.55 * dl_avg + 0.45 * dim_avg, 1)
    else:
        overall_score = round(dim_avg, 1)

    return {
        "overall_score": overall_score,
        "cnn_prediction": cnn_res,
        "bilstm_prediction": bilstm_res,
        "dimension_scores": dims,
        "missing_instructions": dim_report["missing_instructions"],
        "weaknesses": dim_report["weaknesses"],
        "recommendations": dim_report["recommendations"],
    }


def deep_learning_evaluation_node(state: PromptOptimizationState) -> PromptOptimizationState:
    """LangGraph Agent 2: Evaluates original prompt using CNN, Bi-LSTM, and 4 quality dimensions."""
    prompt = state.get("original_prompt", "")
    assessment = evaluate_prompt_intelligence(prompt)

    trace = list(state.get("agent_trace", []))
    trace.append(
        {
            "agent": "Deep Learning Evaluation Agent",
            "status": "completed",
            "summary": (
                f"Overall Score: {assessment['overall_score']}/100 | "
                f"CNN: {assessment['cnn_prediction']['label']} "
                f"({assessment['cnn_prediction']['confidence']}%) | "
                f"Bi-LSTM: {assessment['bilstm_prediction']['label']} "
                f"({assessment['bilstm_prediction']['confidence']}%)"
            ),
            "details": {
                "dimension_scores": assessment["dimension_scores"],
                "weaknesses_count": len(assessment["weaknesses"]),
            },
        }
    )

    existing_missing = state.get("missing_instructions", [])
    combined_missing = list(dict.fromkeys(existing_missing + assessment["missing_instructions"]))

    return {
        "original_score": assessment["overall_score"],
        "cnn_prediction": assessment["cnn_prediction"],
        "bilstm_prediction": assessment["bilstm_prediction"],
        "dimension_scores": assessment["dimension_scores"],
        "weaknesses": assessment["weaknesses"],
        "missing_instructions": combined_missing,
        "recommendations": assessment["recommendations"],
        "agent_trace": trace,
    }