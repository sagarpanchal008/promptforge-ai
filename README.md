# Intelligent Prompt Optimization Assistant 2.0
**An AI-Powered Prompt Intelligence, Evaluation, Security, and Optimization Platform**

## Overview
Intelligent Prompt Optimization Assistant 2.0 transforms raw user prompts into structured, high-reliability instructions using a hybrid **Deep Learning (1D-CNN + Bi-LSTM)** quality engine and a **7-agent autonomous LangGraph workflow** powered by Google Gemini.

## Platform Capabilities
1. **Advanced Prompt Intelligence Engine:** Evaluates prompts using trained 1D-CNN and Bidirectional LSTM classifiers alongside 4-dimensional scoring (*Clarity*, *Specificity*, *Context*, *Completeness*) and missing-instruction detection.
2. **Autonomous Multi-Agent System (LangGraph):** Coordinates 7 specialized agents (*Prompt Analyzer*, *Deep Learning Evaluator*, *Prompt Security Gatekeeper*, *Strategy Selector*, *Gemini Optimizer*, *Quality Reviewer* with bounded retry loops, and *Final Response Synthesizer*).
3. **AI Playground & A/B Testing:** Generates competing prompt variants, executes live LLM comparisons, and scores outputs across *Relevance*, *Completeness*, and *Instruction-Following*.
4. **Hybrid Prompt Security Scanner:** Combines deterministic regex checks and Gemini semantic auditing to detect prompt injection, instruction overrides, jailbreaks, and credential/PII exposure.
5. **Smart Prompt Library & Version Control:** SQLite-backed collections, revision history, unified diff viewer, one-click version restoration, and JSON/Markdown/CSV export.
6. **Feedback & Analytics Dashboard:** Tracks quality improvements, recurring structural weaknesses, user satisfaction ratings, and exports `JSONL` feedback datasets for future model fine-tuning.

## Five-Member Architecture
- **Member 1 (Deep Learning Engineer):** `models/`, `notebooks/`, `utils/preprocessing.py`, `utils/evaluation.py`, `agents/deep_learning_agent.py`
- **Member 2 (Agentic AI Engineer):** `agents/state.py`, `agents/workflow.py`, and LangGraph agent nodes
- **Member 3 (AI Security & Evaluation Engineer):** `security/scanner.py`, `playground/providers.py`, `playground/ab_testing.py`
- **Member 4 (Backend & Database Engineer):** `database/models.py`, `database/prompt_store.py`, `backend/schemas.py`, `backend/api.py`
- **Member 5 (Frontend & Integration Engineer):** `frontend/styles.py`, `frontend/components.py`, `frontend/app.py`, `main.py`

## Local Setup & Execution
1. Activate the project virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1