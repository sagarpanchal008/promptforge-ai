"""
==============================================================================
FILE: agents/strategy_agent.py
PURPOSE:
    Agent 3: Prompt Strategy Agent.
    Formulates a targeted optimization strategy based on the prompt category,
    detected weaknesses, and selected optimization mode.
    Selects role personas, structural blueprints, constraints, and formatting schemas.
==============================================================================
"""

from typing import Any, Dict, List
from agents.state import PromptAgentState

# Personas mapped by domain
DOMAIN_PERSONAS = {
    "Coding": "Staff Software Architect and Senior Systems Specialist",
    "Research": "Senior Academic Research Fellow and Peer Reviewer",
    "Educational": "Distinguished University Professor and STEM Pedagogist",
    "Business": "Principal Management Consultant (McKinsey / BCG alumna)",
    "Content Generation": "Chief Content Strategist and Editorial Director",
    "General": "Expert Systems Strategist and Master Personal Development Mentor",
}

# Blueprints for the 6 Optimization Modes
MODE_BLUEPRINTS = {
    "Basic": [
        "1. Role Persona: Direct and authoritative domain expert",
        "2. Objective Clarification: Clear single-sentence goal",
        "3. Step-by-Step Instructions: Sequenced numbered guidance",
        "4. Expected Output Format: Explicit formatting instructions",
    ],
    "Advanced": [
        "1. Persona & Context: Establish industry background and objective",
        "2. Technical Requirements: Granular scope and boundary limits",
        "3. Phased Execution Roadmap: Numbered procedural steps",
        "4. Negative Constraints: Explicitly list what to avoid",
        "5. Output Deliverable: Tabulated or markdown-formatted response",
    ],
    "Expert": [
        "1. Executive Persona: Top 1% domain authority with rigorous standards",
        "2. Architectural Rationale: Theoretical or business justification",
        "3. Multi-Stage Protocol: Granular end-to-end procedural workflow",
        "4. Edge Cases & Risk Safeguards: Comprehensive boundary resilience",
        "5. Verification Criteria: Executable test cases or review rubric",
        "6. Structured Deliverable: Production-ready markdown template",
    ],
    "Coding": [
        "1. Developer Persona: Staff Software Engineer and Clean Code Evangelist",
        "2. Technical Specifications: Language, runtime version, typing, and SOLID principles",
        "3. Defensive Architecture: Null-checks, boundary validation, and exception logging",
        "4. Performance Constraints: Big-O time and space optimization",
        "5. Production Code: Fully typed, docstring-annotated implementation",
        "6. Verification Suite: 3 runnable unit tests (nominal, edge, failure)",
    ],
    "Research": [
        "1. Scholar Persona: Academic Investigator and Methodologist",
        "2. Theoretical Framework: Foundational literature mechanisms and equations",
        "3. Methodological Taxonomy: Markdown matrix comparing top state-of-the-art approaches",
        "4. Critical Analysis: Unresolved contradictions and validity threats in modern studies",
        "5. Future Directions: Testable experimental hypotheses",
        "6. Academic Tone: Scholarly prose adhering to IEEE/Nature standards",
    ],
    "Educational": [
        "1. Educator Persona: University Professor skilled in cognitive pedagogy",
        "2. Target Learner Definition: Explicit audience background level",
        "3. Intuitive Analogy: Real-world mental model anchor",
        "4. Step-by-Step Mechanism: Sequential breakdown of the phenomenon",
        "5. Misconception Busting: 3 common student pitfalls and clarifications",
        "6. Assessment Checkpoint: 5-question multiple choice quiz with answer key",
    ],
}


def plan_optimization_strategy(state: PromptAgentState) -> Dict:
    """
    Agent 3 Node: Synthesizes analysis and DL evaluations into an optimization plan.
    """
    domain = state.get("detected_domain", "General")
    mode = state.get("optimization_mode", "Advanced")
    weaknesses = state.get("detected_weaknesses", [])
    scores = state.get("quality_scores", {})

    # Select appropriate persona
    persona = DOMAIN_PERSONAS.get(domain, DOMAIN_PERSONAS["General"])

    # Select blueprint based on mode
    mode_key = mode if mode in MODE_BLUEPRINTS else "Advanced"
    blueprint = MODE_BLUEPRINTS[mode_key]

    # Specific repair rules based on detected deficiencies
    repair_rules = []
    if scores.get("clarity", 10) < 6.0:
        repair_rules.append("Inject crisp, unambiguous directive phrasing; remove vague pronouns")
    if scores.get("context", 10) < 6.0:
        repair_rules.append(f"Anchor the prompt in realistic enterprise context tailored for {domain}")
    if scores.get("specificity", 10) < 6.0:
        repair_rules.append("Define explicit technical parameters, inputs, outputs, and boundary conditions")
    if scores.get("completeness", 10) < 6.0:
        repair_rules.append("Add structured deliverable template (Markdown headers, tables, code blocks)")

    strategy = {
        "persona": persona,
        "optimization_mode": mode,
        "blueprint_blocks": blueprint,
        "repair_rules": repair_rules,
        "target_format": "Markdown Hierarchical Document with Bold Labels and Structured Sections",
    }

    return {"optimization_strategy": strategy}


if __name__ == "__main__":
    test_state = {
        "detected_domain": "Coding",
        "optimization_mode": "Coding",
        "quality_scores": {"clarity": 4.0, "context": 3.0, "specificity": 4.5, "completeness": 3.5},
    }
    res = plan_optimization_strategy(test_state)
    print("Agent 3 Strategy:", res)
