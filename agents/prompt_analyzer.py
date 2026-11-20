"""
==============================================================================
FILE: agents/prompt_analyzer.py
PURPOSE:
    Agent 1: Prompt Analyzer.
    Receives raw user prompts, dissects linguistic intent, classifies domain,
    and identifies structural deficiencies (missing context, role, constraints,
    output formatting).
==============================================================================
"""

import re
from typing import Dict, List
from agents.state import PromptAgentState


def analyze_prompt(state: PromptAgentState) -> Dict:
    """
    Agent 1 Node: Ingests the raw prompt, diagnoses objectives, and flags missing components.
    """
    prompt = state.get("original_prompt", "").strip()
    category = state.get("category", "General")
    mode = state.get("optimization_mode", "Advanced")

    word_count = len(prompt.split())
    lower_prompt = prompt.lower()

    # 1. Determine user intent / task objective dynamically
    if any(k in lower_prompt for k in ["email", "memo", "client", "proposal", "business", "roi", "stakeholder", "delay", "budget", "rfp"]):
        intent = "Executive / Corporate Communication"
        detected_domain = "Business"
    elif any(k in lower_prompt for k in ["teach", "explain", "student", "learn", "concept", "lesson", "undergraduate", "curriculum", "pedagogy"]):
        intent = "Pedagogical Explanation / Educational Instruction"
        detected_domain = "Educational"
    elif any(k in lower_prompt for k in ["research", "paper", "literature", "study", "analysis", "academic", "taxonomy", "hypothesis", "survey"]):
        intent = "Academic Investigation / Literature Synthesis"
        detected_domain = "Research"
    elif any(k in lower_prompt for k in ["blog", "article", "post", "social media", "story", "copy", "newsletter", "headline"]):
        intent = "Creative Content / Copywriting"
        detected_domain = "Content Generation"
    elif any(k in lower_prompt for k in ["code", "script", "function", "bug", "python", "sql", "api", "docker", "class", "cache", "algorithm", "endpoint", "pytest"]):
        intent = "Software Engineering / Technical Implementation"
        detected_domain = "Coding"
    else:
        intent = "General Task / Direct Query"
        detected_domain = "General"

    # Use user category ONLY if explicitly set and not Auto-Detect/General, otherwise smart detected domain
    if category in ["Auto-Detect (Smart)", "Auto-Detect", "General", ""] or category is None:
        final_category = detected_domain
    else:
        # If user selected a specific category, verify if prompt strongly conflicts (e.g. Email prompt with Coding selected)
        if category == "Coding" and detected_domain in ["Business", "Educational", "Research"] and not any(k in lower_prompt for k in ["code", "function", "script", "python"]):
            final_category = detected_domain  # Smart override to avoid nonsensical Coding template for emails
        else:
            final_category = category

    # 2. Detect missing structural components
    missing = []

    # Persona Check
    if not any(k in lower_prompt for k in ["act as", "you are", "role", "expert", "as a", "from the perspective"]):
        missing.append("Missing Role Persona (no expert identity defined)")

    # Target Audience Check
    if not any(k in lower_prompt for k in ["for beginners", "for students", "for executive", "for client", "audience", "tailored for"]):
        missing.append("Missing Target Audience / Stakeholder Context")

    # Constraints Check
    if not any(k in lower_prompt for k in ["limit", "constraint", "avoid", "must", "strictly", "time complexity", "budget"]):
        missing.append("Missing Technical Constraints / Guardrails")

    # Output Format Check
    if not any(k in lower_prompt for k in ["json", "markdown", "table", "step-by-step", "bullet", "format as", "csv", "summary"]):
        missing.append("Missing Explicit Output Format Requirements")

    # Examples Check
    if not any(k in lower_prompt for k in ["example", "sample", "test case", "demonstration", "analogy"]):
        missing.append("Missing Few-Shot Examples or Verification Cases")

    # Length deficiency warning
    if word_count < 15:
        missing.append("Critical Brevity: Prompt lacks necessary task specification")

    return {
        "analyzed_intent": intent,
        "detected_domain": final_category,
        "missing_components": missing,
        "input_word_count": word_count,
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 2),
    }


if __name__ == "__main__":
    test_state = {
        "original_prompt": "write a python script to clean data",
        "category": "Coding",
        "optimization_mode": "Coding",
    }
    res = analyze_prompt(test_state)
    print("Agent 1 Output:", res)
