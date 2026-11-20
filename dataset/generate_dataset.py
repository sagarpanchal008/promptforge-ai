"""
==============================================================================
FILE: dataset/generate_dataset.py
PURPOSE:
    Generates a reproducible, high-quality, balanced dataset of user prompts
    across 6 domains and 3 quality categories (Poor, Average, Good).
    Includes quality metric scores, identified weaknesses, and gold-standard
    optimized prompt counterparts.

HOW IT CONNECTS TO THE PROJECT:
    - Creates 'dataset/prompts.csv', the foundational raw dataset.
    - Supplies training and testing data for Phase 2 (Preprocessing),
      Phase 3 (CNN Model), and Phase 4 (LSTM Model).
    - Informs Agent 1 (Prompt Analyzer) and Agent 2 (DL Evaluator) in Phase 6.

EXECUTION:
    python dataset/generate_dataset.py
==============================================================================
"""

import csv
import os
import random
import sys
from typing import Dict, List, Tuple

# Set random seed for 100% reproducibility across all runs
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Output directory and file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_CSV = os.path.join(DATASET_DIR, "prompts.csv")

# Categories required by the project specifications
CATEGORIES = [
    "Educational",
    "Coding",
    "Research",
    "Content Generation",
    "Business",
    "General",
]

# Quality labels for classification
QUALITY_LABELS = ["Poor", "Average", "Good"]

# -----------------------------------------------------------------------------
# SEED DOMAIN DATA & VOCABULARY BANKS
# -----------------------------------------------------------------------------
# These banks define real-world subjects, tasks, constraints, and personas
# for each of the 6 prompt categories.

DOMAIN_SEEDS = {
    "Educational": {
        "topics": [
            "Photosynthesis and Cellular Respiration",
            "Quantum Superposition and Entanglement",
            "Fourier Transform in Signal Processing",
            "The French Revolution Causes and Impacts",
            "Binary Search Trees and AVL Balancing",
            "Plate Tectonics and Continental Drift",
            "Mendelian Genetics and Punnett Squares",
            "Central Limit Theorem in Statistics",
            "Supply and Demand Elasticity in Microeconomics",
            "Newtonian Mechanics and Conservation of Momentum",
        ],
        "personas": [
            "University Professor in STEM",
            "High School Science Educator",
            "Interactive Pedagogical Tutor",
            "Academic Curriculum Developer",
        ],
        "audiences": [
            "first-year undergraduate students",
            "10th-grade high school learners",
            "curious beginners with no prior technical background",
            "advanced self-taught hobbyists",
        ],
        "poor_templates": [
            "explain {topic}",
            "tell me about {topic}",
            "what is {topic} and why does it matter",
            "teach me {topic} quickly",
            "summary of {topic}",
            "give info on {topic} for exam",
            "write about {topic} please",
        ],
        "average_templates": [
            "Explain {topic} in simple words for {audience}. Include a basic definition and one real-world example.",
            "Write a study guide explaining {topic}. Cover the core concepts and list 3 common student misconceptions.",
            "Please teach me {topic}. Explain how it works step-by-step and provide a brief summary at the end.",
            "Create a lesson overview on {topic} suitable for {audience}. Describe the main mechanism and significance.",
            "Explain the principles behind {topic} with an illustrative analogy and a summary paragraph.",
        ],
        "good_templates": [
            (
                "Act as an expert {persona}. Provide a comprehensive, pedagogy-backed masterclass on {topic} "
                "tailored specifically for {audience}. Structure your explanation into the following distinct sections:\n"
                "1. Intuitive Real-World Analogy (to anchor conceptual intuition)\n"
                "2. Formal Mathematical / Theoretical Mechanics (with key definitions)\n"
                "3. Step-by-Step Practical Demonstration or Case Study\n"
                "4. 3 Frequently Confused Concepts and Their Clarifications\n"
                "5. A 5-question Self-Assessment Multiple Choice Quiz with an Answer Key.\n"
                "Format using clear Markdown headers, bold terminology, and bullet points."
            ),
            (
                "You are an acclaimed {persona}. Write an exhaustive instructional guide on {topic} for {audience}. "
                "Ensure maximum clarity and rigor. Include:\n"
                "- Executive Summary: 2-sentence distillation of the phenomenon.\n"
                "- Core Principles: Break down the underlying mechanics into 4 sequential stages.\n"
                "- Visualizable Scenario: Walk through a concrete mental experiment or diagrammatic description.\n"
                "- Practical Application: How this applies in modern industry or everyday life.\n"
                "- Review Exercises: 3 thought-provoking conceptual questions with detailed explanatory solutions.\n"
                "Maintain an encouraging, highly educational tone throughout."
            ),
        ],
        "poor_weaknesses": "Lacks target audience, No pedagogical structure, Missing examples, Ambiguous depth, No formatting constraints",
        "avg_weaknesses": "Missing structural scaffolding, Lacks assessment checkpoints, Underspecified tone, Absence of role persona",
        "good_weaknesses": "None (Well-structured prompt with persona, audience, constraints, and explicit output format)",
    },
    "Coding": {
        "topics": [
            ("Python", "pandas DataFrame duplicate cleaning with multi-index reset"),
            ("FastAPI", "asynchronous JWT authentication middleware with rate limiting"),
            ("React TypeScript", "accessible dark/light theme toggle component with local storage persistence"),
            ("SQL", "recursive CTE query to traverse hierarchical employee-manager tables"),
            ("Docker", "multi-stage build pipeline for a Node.js microservice minimizing image size"),
            ("PyTorch", "custom loss function implementing Focal Loss for imbalanced classification"),
            ("Go", "concurrent worker pool pattern with channels and context cancellation"),
            ("C++", "thread-safe lock-free ring buffer for low-latency audio processing"),
            ("Rust", "zero-copy string parser with custom error handling via Result types"),
            ("TensorFlow", "custom Keras Callback calculating validation macro F1-score at epoch end"),
        ],
        "personas": [
            "Staff Software Architect and Clean Code Evangelist",
            "Senior Backend Engineer and Systems Specialist",
            "Principal Full-Stack Developer",
            "Lead ML Infrastructure Engineer",
        ],
        "poor_templates": [
            "write code for {topic}",
            "fix {lang} code",
            "how to do {topic} in {lang}",
            "{lang} script for {topic} fast",
            "create function {topic}",
            "give me {lang} code for this problem",
        ],
        "average_templates": [
            "Write a {lang} function that implements {topic}. Include comments and handle basic error cases.",
            "How do I build {topic} using {lang}? Show a working code snippet and explain how it runs.",
            "Write clean {lang} code to solve: {topic}. Make sure to add proper variable naming and docstrings.",
            "Provide a working code example in {lang} for {topic}. Explain the main logic in 3-4 bullet points.",
            "Create a modular script in {lang} that handles {topic}. Handle unexpected inputs gracefully.",
        ],
        "good_templates": [
            (
                "Act as a {persona}. Write a production-ready, highly robust implementation in {lang} for {topic}.\n"
                "Adhere strictly to the following technical specifications:\n"
                "1. Architecture: Modular, adhering to SOLID principles and idiomatic {lang} conventions.\n"
                "2. Type Safety & Validation: Include comprehensive type annotations and defensive input validation.\n"
                "3. Edge Cases & Error Handling: Gracefully manage boundary conditions, null inputs, and exceptions.\n"
                "4. Performance Constraints: Optimize for O(N) time and minimal memory allocation.\n"
                "5. Documentation: Provide complete docstrings (Sphinx/PEP 257 style) and inline comments.\n"
                "6. Verification: Supply 3 executable unit tests (covering happy path, invalid input, and boundary limit)."
            ),
            (
                "You are a {persona}. Implement an enterprise-grade solution in {lang} for {topic}.\n"
                "Your response must include:\n"
                "- Architecture Overview: Brief justification of chosen data structures and design patterns.\n"
                "- Production Code: Fully typed, clean code with zero external dependencies unless strictly needed.\n"
                "- Security & Resilience: Explicit input sanitation and exception logging.\n"
                "- Comprehensive Test Suite: Pytest / test suite verifying nominal and failure scenarios.\n"
                "- Usage Demonstration: Runnable minimal reproducible example with sample console output."
            ),
        ],
        "poor_weaknesses": "No language version specified, Missing edge-case requirements, No error handling constraints, Undefined inputs/outputs, Lacks testing guidelines",
        "avg_weaknesses": "Missing explicit error contracts, Lacks test suite specifications, Absence of performance constraints, No role definition",
        "good_weaknesses": "None (Comprehensive prompt with technical constraints, typing, error handling, and test specifications)",
    },
    "Research": {
        "topics": [
            "Transformer Self-Attention vs State Space Models (Mamba) efficiency trade-offs",
            "CRISPR-Cas9 off-target mutation mitigation mechanisms in vivo",
            "Perovskite solar cell degradation kinetics under thermal and moisture stress",
            "Federated Learning privacy preservation techniques under Byzantine poisoning attacks",
            "Retrieval-Augmented Generation (RAG) hallucination detection benchmarks",
            "Solid-state lithium battery electrolyte ionic conductivity optimization",
            "Epigenetic methylation biomarkers for early colorectal carcinoma detection",
            "Zero-Knowledge Rollups scalability compared to Optimistic Rollups in Ethereum",
            "Reinforcement Learning with Human Feedback (RLHF) reward collapse vulnerabilities",
            "Microplastic bioaccumulation in marine benthic trophic networks",
        ],
        "personas": [
            "Senior Research Fellow and Peer Reviewer",
            "Principal Scientist in Computational Sciences",
            "Postdoctoral Academic Investigator",
            "Systematic Review Methodologist",
        ],
        "poor_templates": [
            "summarize research on {topic}",
            "literature review {topic}",
            "what are papers saying about {topic}",
            "research summary {topic} write 500 words",
            "find pros and cons of {topic}",
            "write paper introduction on {topic}",
        ],
        "average_templates": [
            "Write a structured literature summary on {topic}. Highlight the main breakthroughs and current challenges.",
            "Summarize the recent scientific consensus on {topic}. Compare the leading methodological approaches.",
            "Provide an academic overview of {topic}. Mention 3 key foundational studies and their primary findings.",
            "Write a research synthesis on {topic}. Discuss the theoretical foundations and open experimental questions.",
            "Analyze current research trends in {topic}. List the major advantages, limitations, and future outlook.",
        ],
        "good_templates": [
            (
                "Act as a {persona}. Conduct a rigorous, critical literature review and comparative analysis on {topic}.\n"
                "Structure your academic synthesis into the following formal sections:\n"
                "1. Theoretical Framework & Foundational Mechanisms: Core physics/algorithmic equations governing the domain.\n"
                "2. Methodological Taxonomy: A markdown comparison table contrasting the top 3 state-of-the-art approaches across:\n"
                "   - Computational Complexity / Throughput\n"
                "   - Empirical Reliability & Error Margins\n"
                "   - Scalability and Implementation Bottlenecks\n"
                "3. Critical Vulnerabilities & Contradictions: Unresolved debates and methodological pitfalls in recent literature.\n"
                "4. Future Research Directions: 3 high-impact, testable experimental hypotheses for future investigation.\n"
                "Maintain an objective, scholarly tone adhering to IEEE/Nature review standards."
            ),
            (
                "You are an esteemed {persona}. Write a comprehensive research synthesis on {topic}.\n"
                "Your analysis must address:\n"
                "- Abstract: High-density 150-word synthesis of current paradigm shifts.\n"
                "- Deep-Dive Mechanistic Breakdown: How contemporary findings challenge classical assumptions.\n"
                "- Quantitative Comparison Matrix: Tabulated evaluation of performance benchmarks and experimental datasets.\n"
                "- Limitations & Threat to Validity: Scrutiny of measurement artifacts and confounding variables.\n"
                "- Methodological Recommendations: Actionable protocol guidelines for laboratory or computational replication.\n"
                "Adopt rigorous academic prose suitable for graduate-level investigators."
            ),
        ],
        "poor_weaknesses": "Lacks academic rigor, Missing methodological criteria, No comparison framework, Absence of citation/evidence standards",
        "avg_weaknesses": "Lacks quantitative evaluation criteria, Missing critical vulnerability analysis, Underspecified comparative matrix",
        "good_weaknesses": "None (High-rigor academic prompt with taxonomy, comparative matrix, hypotheses, and structured sections)",
    },
    "Content Generation": {
        "topics": [
            ("B2B SaaS Founder", "Transitioning from monolithic legacy infrastructure to cloud-native microservices"),
            ("Health & Wellness Brand", "Evidence-based benefits of circadian rhythm alignment for cognitive longevity"),
            ("Fintech Startup", "Demystifying decentralized automated market makers for retail investors"),
            ("Renewable Energy Advocate", "The economic tipping point of commercial rooftop solar and battery storage"),
            ("Cybersecurity Consultant", "Zero Trust Architecture implementation roadmap for distributed remote teams"),
            ("EdTech Pioneer", "How generative AI is reshaping personalized asynchronous learning curves"),
            ("Productivity Coach", "Time-blocking vs Energy-management: Overcoming executive burnout"),
            ("Urban Sustainability Planner", "Smart cities and IoT-driven urban storm-water management"),
        ],
        "personas": [
            "Chief Content Strategist and Thought Leader",
            "Senior Editorial Director and Brand Storyteller",
            "Award-winning Tech Columnist",
            "Growth Marketing Copywriter",
        ],
        "poor_templates": [
            "write blog post about {topic}",
            "write an article on {topic}",
            "create social media post for {topic}",
            "write 1000 words on {topic}",
            "make marketing copy about {topic}",
            "write content for {author}: {topic}",
        ],
        "average_templates": [
            "Write an engaging 800-word blog post from the perspective of a {author} on the topic: '{topic}'. Include an intro, 3 body points, and a conclusion.",
            "Create an informative LinkedIn article discussing '{topic}'. Aim it at working professionals and include a call to action.",
            "Write a comprehensive thought leadership piece on '{topic}'. Use a persuasive tone and provide practical takeaways.",
            "Draft a newsletter issue for modern professionals exploring '{topic}'. Include subheadings and bullet points for readability.",
            "Compose a compelling article on '{topic}' written for industry stakeholders. Conclude with forward-looking industry advice.",
        ],
        "good_templates": [
            (
                "Act as a {persona}. Craft an authoritative, highly engaging thought-leadership article from the voice of a {author} "
                "addressing: '{topic}'.\n"
                "Follow this strict editorial blueprint:\n"
                "1. Hook: Begin with a compelling contrarian statistic or counter-intuitive narrative opening (no clichéd rhetorical questions).\n"
                "2. Core Thesis: Clearly articulate the paradigm shift occurring in the industry within 2 concise paragraphs.\n"
                "3. Strategic Deep-Dive: Present 3 actionable pillars substantiated by real-world enterprise case studies and quantitative metrics.\n"
                "4. Implementation Roadmap: Provide a 4-step execution framework formatted in a clear Markdown table.\n"
                "5. Strategic Conclusion & CTA: Synthesize long-term strategic implications and prompt readers with an engaging discussion prompt.\n"
                "Tone: Authoritative, pragmatic, and inspiring. Avoid buzzwords and marketing fluff."
            ),
            (
                "You are an acclaimed {persona}. Write a high-converting, insightful pillar article for a {author} on: '{topic}'.\n"
                "Content Requirements:\n"
                "- Target Audience: C-suite executives, senior decision-makers, and technical directors.\n"
                "- Narrative Arc: Problem-Agitate-Solve framework with crisp storytelling.\n"
                "- Structural Elements: Compelling H2/H3 subheadings, quote callouts, and numbered tactical checklists.\n"
                "- Data-Driven Evidence: Integrate realistic industry benchmarks and quantifiable outcomes (e.g., % latency reduction, ROI timeline).\n"
                "- Actionable Deliverable: A downloadable summary checklist blueprint at the end.\n"
                "Length: Exactly 1,000 words. Prose style: Sophisticated, crisp, and actionable."
            ),
        ],
        "poor_weaknesses": "No target audience specified, Missing word count/length guideline, Undefined brand tone, Absence of storytelling structure",
        "avg_weaknesses": "Lacks specific hook framework, Underspecified data evidence requirements, Generic call-to-action guidelines",
        "good_weaknesses": "None (Excellently structured editorial prompt with tone, target audience, structural blueprint, and explicit formatting)",
    },
    "Business": {
        "topics": [
            ("Enterprise ERP migration proposal", "Chief Information Officer and Board of Directors", "$1.2M capital allocation"),
            ("Product launch go-to-market strategy", "VP of Product and Marketing Leads", "B2B AI analytics platform"),
            ("Quarterly financial performance review", "Institutional Investors and Shareholders", "Q3 revenue shortfall mitigation"),
            ("Client contract renegotiation email", "Enterprise Procurement Director", "15% annual service fee indexation"),
            ("Vendor vendor evaluation request (RFP)", "Supply Chain Operations Lead", "Global cold-chain logistics provider"),
            ("Employee return-to-office transition plan", "All-hands company distribution", "Hybrid 3-day flexible framework"),
            ("Customer churn mitigation action plan", "Customer Success Team", "SaaS retention stabilization campaign"),
            ("New market expansion feasibility brief", "Executive Committee", "Southeast Asia fintech penetration"),
        ],
        "personas": [
            "Management Consultant (McKinsey / BCG alumna)",
            "Chief Strategy Officer and Corporate Advisor",
            "Executive Communications Specialist",
            "Senior Commercial Director",
        ],
        "poor_templates": [
            "write business proposal for {task}",
            "email client about {task}",
            "make business plan for {task}",
            "write memo to {stakeholder}",
            "prepare presentation on {task}",
            "help me write a letter for {task}",
        ],
        "average_templates": [
            "Draft a professional business memo regarding {task} addressed to {stakeholder}. Clearly explain the context and our proposal.",
            "Write a formal corporate proposal on {task} for {stakeholder}. Outline key objectives, timeline, and projected budget considerations: {context}.",
            "Compose a persuasive business email to {stakeholder} discussing {task}. Maintain a professional and diplomatic tone.",
            "Create an executive summary for {task}. Target {stakeholder} and include 3 main recommendations.",
            "Write a business justification document for {task} highlighting key benefits and risk mitigation strategies.",
        ],
        "good_templates": [
            (
                "Act as an elite {persona}. Draft an executive-ready, highly persuasive business proposal for: '{task}'.\n"
                "Primary Audience: {stakeholder}.\n"
                "Key Context & Parameters: {context}.\n"
                "Structure the document according to rigorous corporate standards:\n"
                "1. Executive Summary: 1-page synthesis following the Minto Pyramid Principle (Situation, Complication, Solution, Impact).\n"
                "2. Strategic & Financial Rationale: Quantifiable business ROI, productivity gains, and alignment with enterprise OKRs.\n"
                "3. Phased Implementation Roadmap: Milestone schedule (Phase 1 to 4) detailing deliverables, RACI matrix, and timeline.\n"
                "4. Comprehensive Risk Matrix: Markdown table detailing 4 primary risks (Operational, Financial, Technical, Compliance), "
                "their probability/impact rating, and mitigation protocols.\n"
                "5. Immediate Actionable Next Steps: Explicit sign-off requirements and milestone kickoff date.\n"
                "Tone: Polished, diplomatic, data-driven, and commercially astute."
            ),
            (
                "You are an experienced {persona}. Compose a high-stakes executive communication regarding: '{task}'.\n"
                "Addressed to: {stakeholder}.\n"
                "Contextual details: {context}.\n"
                "Strict requirements:\n"
                "- Objective: Align stakeholders and secure formal executive sponsorship.\n"
                "- Structure: Context setting -> Financial & operational justification -> Implementation hurdles & countermeasures -> Decision deadline.\n"
                "- Format: Formal corporate memo with numbered paragraphs, executive callout boxes, and tabulated milestones.\n"
                "- Communication Tone: Confident, respectful, unambiguous, and risk-conscious."
            ),
        ],
        "poor_weaknesses": "Lacks stakeholder context, Missing financial/business metrics, No structured proposal framework, Absent risk assessment",
        "avg_weaknesses": "Missing risk mitigation matrix, Underspecified ROI/financial metrics, Lacks explicit milestone/RACI framework",
        "good_weaknesses": "None (Flawlessly structured business prompt using Minto Pyramid, risk matrix, RACI, and actionable ROI metrics)",
    },
    "General": {
        "topics": [
            "Planning a 7-day budget travel itinerary in Japan with cultural etiquette tips",
            "Designing a balanced whole-food plant-based weekly meal prep schedule",
            "Setting up a productive minimalist home office workspace for ergonomic wellness",
            "Formulating a disciplined personal finance budget using the 50/30/20 rule",
            "Developing an effective morning routine to improve deep focus and sleep quality",
            "Organizing a community volunteer charity fundraising drive from scratch",
            "Building a comprehensive emergency preparedness family safety kit",
            "Negotiating apartment lease terms and security deposit protections with landlords",
        ],
        "personas": [
            "Expert Lifestyle Strategist and Systems Organizer",
            "Master Personal Development Mentor",
            "Professional Logistics & Planning Consultant",
        ],
        "poor_templates": [
            "plan my trip {topic}",
            "give me advice on {topic}",
            "how to do {topic}",
            "help with {topic} please",
            "make a plan for {topic}",
            "tips on {topic}",
        ],
        "average_templates": [
            "Create a practical, step-by-step guide for: {topic}. Provide actionable tips and a daily checklist.",
            "I need comprehensive advice on: {topic}. Give me 5 proven recommendations and explain why they work.",
            "Write an organized planning framework for {topic}. Include common mistakes to avoid and a timeline.",
            "How should I approach {topic}? Break your answer down into preparation, execution, and review stages.",
            "Provide a realistic, budget-conscious plan for {topic}. Include actionable steps for a beginner.",
        ],
        "good_templates": [
            (
                "Act as a {persona}. Develop an exhaustive, turnkey operational master plan for: '{topic}'.\n"
                "To ensure maximum utility, structure your guide into the following explicit components:\n"
                "1. Scope & Prerequisite Audit: Inventory of necessary tools, realistic budget estimates, and initial setup requirements.\n"
                "2. Chronological Action Blueprint: Phased schedule (Day-by-Day or Stage-by-Stage) with explicit micro-tasks and duration estimates.\n"
                "3. Tabulated Resource & Budget Matrix: Markdown table detailing itemized costs, priority tier (Essential vs Optional), and contingencies.\n"
                "4. Pitfalls & Proactive Safeguards: 4 common execution failures and actionable protocols to preempt them.\n"
                "5. Practical Verification Checklist: Print-ready checklist to evaluate milestone completion.\n"
                "Tone: Methodical, motivating, empathetic, and exceptionally practical."
            ),
            (
                "You are an acclaimed {persona}. Produce an end-to-end actionable masterguide on: '{topic}'.\n"
                "Guide Specifications:\n"
                "- Intended User: Busy individual seeking optimal efficiency without sacrificing quality.\n"
                "- Structure: Executive Overview -> Step-by-Step Blueprint -> Resource Allocation Table -> Contingency Protocols.\n"
                "- Formatting: Clear hierarchical markdown headers, bulleted action items with estimated time commitments, and bold warning callouts.\n"
                "- Output Style: Crisp, prescriptive, and entirely devoid of generic filler advice."
            ),
        ],
        "poor_weaknesses": "Lacks specific constraints, No timeline or budget, Missing step-by-step breakdown, Undefined user context",
        "avg_weaknesses": "Missing budget/resource breakdown, Lacks contingency safeguards, Underspecified timeline commitments",
        "good_weaknesses": "None (Fully specified real-world prompt with prerequisites, phased blueprint, budget matrix, and safeguards)",
    },
}


# -----------------------------------------------------------------------------
# GENERATION ENGINE
# -----------------------------------------------------------------------------

def generate_sample_for_category(category: str, quality: str, sample_idx: int) -> Dict:
    """
    Generates a realistic, diverse prompt sample for a given category and quality level.

    Args:
        category: One of ['Educational', 'Coding', 'Research', 'Content Generation', 'Business', 'General']
        quality: One of ['Poor', 'Average', 'Good']
        sample_idx: Integer counter for unique ID generation

    Returns:
        Dict representing one complete dataset row.
    """
    data = DOMAIN_SEEDS[category]

    # Select random items from the category vocabulary bank
    persona = random.choice(data["personas"])

    if category == "Educational":
        topic = random.choice(data["topics"])
        audience = random.choice(data["audiences"])
        if quality == "Poor":
            prompt = random.choice(data["poor_templates"]).format(topic=topic)
            scores = (
                random.uniform(1.5, 3.5),  # clarity
                random.uniform(1.0, 2.5),  # context
                random.uniform(1.0, 3.0),  # specificity
                random.uniform(1.0, 2.5),  # completeness
            )
            weaknesses = data["poor_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, topic=topic, audience=audience
            )
        elif quality == "Average":
            prompt = random.choice(data["average_templates"]).format(
                topic=topic, audience=audience
            )
            scores = (
                random.uniform(5.0, 6.8),
                random.uniform(4.5, 6.5),
                random.uniform(4.8, 6.7),
                random.uniform(4.5, 6.5),
            )
            weaknesses = data["avg_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, topic=topic, audience=audience
            )
        else:  # Good
            prompt = random.choice(data["good_templates"]).format(
                persona=persona, topic=topic, audience=audience
            )
            scores = (
                random.uniform(8.5, 9.8),
                random.uniform(8.3, 9.7),
                random.uniform(8.6, 9.9),
                random.uniform(8.5, 9.8),
            )
            weaknesses = data["good_weaknesses"]
            optimized = prompt  # Already optimal

    elif category == "Coding":
        lang, topic = random.choice(data["topics"])
        if quality == "Poor":
            prompt = random.choice(data["poor_templates"]).format(lang=lang, topic=topic)
            scores = (
                random.uniform(2.0, 3.8),
                random.uniform(1.0, 2.5),
                random.uniform(1.5, 3.2),
                random.uniform(1.0, 2.5),
            )
            weaknesses = data["poor_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, lang=lang, topic=topic
            )
        elif quality == "Average":
            prompt = random.choice(data["average_templates"]).format(
                lang=lang, topic=topic
            )
            scores = (
                random.uniform(5.5, 7.0),
                random.uniform(4.5, 6.6),
                random.uniform(5.2, 7.0),
                random.uniform(4.8, 6.8),
            )
            weaknesses = data["avg_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, lang=lang, topic=topic
            )
        else:  # Good
            prompt = random.choice(data["good_templates"]).format(
                persona=persona, lang=lang, topic=topic
            )
            scores = (
                random.uniform(8.7, 9.9),
                random.uniform(8.5, 9.8),
                random.uniform(8.8, 10.0),
                random.uniform(8.6, 9.9),
            )
            weaknesses = data["good_weaknesses"]
            optimized = prompt

    elif category == "Research":
        topic = random.choice(data["topics"])
        if quality == "Poor":
            prompt = random.choice(data["poor_templates"]).format(topic=topic)
            scores = (
                random.uniform(1.5, 3.5),
                random.uniform(1.0, 2.2),
                random.uniform(1.2, 3.0),
                random.uniform(1.0, 2.4),
            )
            weaknesses = data["poor_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, topic=topic
            )
        elif quality == "Average":
            prompt = random.choice(data["average_templates"]).format(topic=topic)
            scores = (
                random.uniform(5.2, 6.8),
                random.uniform(4.8, 6.7),
                random.uniform(5.0, 6.8),
                random.uniform(4.6, 6.5),
            )
            weaknesses = data["avg_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, topic=topic
            )
        else:  # Good
            prompt = random.choice(data["good_templates"]).format(
                persona=persona, topic=topic
            )
            scores = (
                random.uniform(8.6, 9.8),
                random.uniform(8.4, 9.7),
                random.uniform(8.7, 9.9),
                random.uniform(8.5, 9.8),
            )
            weaknesses = data["good_weaknesses"]
            optimized = prompt

    elif category == "Content Generation":
        author, topic = random.choice(data["topics"])
        if quality == "Poor":
            prompt = random.choice(data["poor_templates"]).format(
                author=author, topic=topic
            )
            scores = (
                random.uniform(2.0, 3.8),
                random.uniform(1.2, 2.6),
                random.uniform(1.5, 3.2),
                random.uniform(1.2, 2.8),
            )
            weaknesses = data["poor_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, author=author, topic=topic
            )
        elif quality == "Average":
            prompt = random.choice(data["average_templates"]).format(
                author=author, topic=topic
            )
            scores = (
                random.uniform(5.4, 7.0),
                random.uniform(4.8, 6.8),
                random.uniform(5.0, 6.9),
                random.uniform(4.9, 6.7),
            )
            weaknesses = data["avg_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, author=author, topic=topic
            )
        else:  # Good
            prompt = random.choice(data["good_templates"]).format(
                persona=persona, author=author, topic=topic
            )
            scores = (
                random.uniform(8.6, 9.9),
                random.uniform(8.5, 9.8),
                random.uniform(8.7, 9.9),
                random.uniform(8.6, 9.8),
            )
            weaknesses = data["good_weaknesses"]
            optimized = prompt

    elif category == "Business":
        task, stakeholder, context = random.choice(data["topics"])
        if quality == "Poor":
            prompt = random.choice(data["poor_templates"]).format(
                task=task, stakeholder=stakeholder
            )
            scores = (
                random.uniform(1.8, 3.6),
                random.uniform(1.0, 2.5),
                random.uniform(1.4, 3.0),
                random.uniform(1.1, 2.6),
            )
            weaknesses = data["poor_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, task=task, stakeholder=stakeholder, context=context
            )
        elif quality == "Average":
            prompt = random.choice(data["average_templates"]).format(
                task=task, stakeholder=stakeholder, context=context
            )
            scores = (
                random.uniform(5.3, 6.9),
                random.uniform(4.9, 6.8),
                random.uniform(5.1, 6.9),
                random.uniform(4.8, 6.7),
            )
            weaknesses = data["avg_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, task=task, stakeholder=stakeholder, context=context
            )
        else:  # Good
            prompt = random.choice(data["good_templates"]).format(
                persona=persona, task=task, stakeholder=stakeholder, context=context
            )
            scores = (
                random.uniform(8.7, 9.9),
                random.uniform(8.6, 9.8),
                random.uniform(8.8, 10.0),
                random.uniform(8.7, 9.9),
            )
            weaknesses = data["good_weaknesses"]
            optimized = prompt

    else:  # General
        topic = random.choice(data["topics"])
        if quality == "Poor":
            prompt = random.choice(data["poor_templates"]).format(topic=topic)
            scores = (
                random.uniform(1.8, 3.7),
                random.uniform(1.0, 2.5),
                random.uniform(1.3, 3.1),
                random.uniform(1.0, 2.5),
            )
            weaknesses = data["poor_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, topic=topic
            )
        elif quality == "Average":
            prompt = random.choice(data["average_templates"]).format(topic=topic)
            scores = (
                random.uniform(5.2, 6.9),
                random.uniform(4.7, 6.7),
                random.uniform(5.0, 6.8),
                random.uniform(4.7, 6.6),
            )
            weaknesses = data["avg_weaknesses"]
            optimized = random.choice(data["good_templates"]).format(
                persona=persona, topic=topic
            )
        else:  # Good
            prompt = random.choice(data["good_templates"]).format(
                persona=persona, topic=topic
            )
            scores = (
                random.uniform(8.5, 9.8),
                random.uniform(8.4, 9.7),
                random.uniform(8.6, 9.9),
                random.uniform(8.5, 9.8),
            )
            weaknesses = data["good_weaknesses"]
            optimized = prompt

    clarity, context_sc, specificity, completeness = [round(s, 2) for s in scores]
    overall = round((clarity + context_sc + specificity + completeness) / 4.0, 2)

    return {
        "prompt_id": f"PRM_{sample_idx:05d}",
        "prompt": prompt.strip(),
        "category": category,
        "clarity_score": clarity,
        "context_score": context_sc,
        "specificity_score": specificity,
        "completeness_score": completeness,
        "overall_score": overall,
        "quality_label": quality,
        "weaknesses": weaknesses,
        "optimized_prompt": optimized.strip(),
    }


def generate_full_dataset(samples_per_slice: int = 70) -> List[Dict]:
    """
    Generates a perfectly balanced dataset.
    6 categories * 3 quality classes * samples_per_slice = Total rows.
    Default: 6 * 3 * 70 = 1,260 realistic prompt instances.
    """
    dataset = []
    sample_id = 1

    print(f"[*] Initiating dataset generation: 6 categories x 3 quality levels...")
    for category in CATEGORIES:
        for quality in QUALITY_LABELS:
            for _ in range(samples_per_slice):
                sample = generate_sample_for_category(category, quality, sample_id)
                dataset.append(sample)
                sample_id += 1

    # Shuffle the dataset to ensure training/validation splits will not be ordered by category
    random.shuffle(dataset)
    return dataset


def save_dataset_to_csv(dataset: List[Dict], output_path: str):
    """
    Writes the generated dataset rows to a standardized CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "prompt_id",
        "prompt",
        "category",
        "clarity_score",
        "context_score",
        "specificity_score",
        "completeness_score",
        "overall_score",
        "quality_label",
        "weaknesses",
        "optimized_prompt",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"[+] Dataset saved successfully to: {output_path}")
    print(f"[+] Total Samples: {len(dataset)}")


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1: PROMPT DATASET GENERATION ENGINE")
    print("=" * 70)
    # Generate 1,260 balanced, realistic prompt instances (420 Poor, 420 Average, 420 Good)
    prompts_data = generate_full_dataset(samples_per_slice=70)
    save_dataset_to_csv(prompts_data, OUTPUT_CSV)
    print("=" * 70)
    print("[SUCCESS] Phase 1 Dataset Generation Complete!")
    print("=" * 70)
