"""
==============================================================================
FILE: main.py
PURPOSE:
    Master Entrypoint for the Intelligent Prompt Optimization Assistant.
    Unified CLI launcher supporting:
    - Interactive CLI Mode
    - Streamlit Web Dashboard Launcher
    - Model Benchmark Evaluation
    - Automated Comprehensive Test Suite

USAGE EXAMPLES:
    python main.py --mode cli
    python main.py --mode streamlit
    python main.py --mode evaluate
    python main.py --mode test
==============================================================================
"""

import argparse
import os
import subprocess
import sys

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.workflow import run_prompt_optimization
from utils.evaluation import run_comparative_evaluation


def run_interactive_cli():
    """Runs interactive command-line prompt optimizer."""
    print("\n" + "=" * 75)
    print("🧠 INTELLIGENT PROMPT OPTIMIZATION ASSISTANT (CLI MODE)")
    print("Deep Learning (1D-CNN + Bi-LSTM) & LangGraph Agentic AI Architecture")
    print("=" * 75)

    while True:
        try:
            print("\nEnter a prompt to analyze & optimize (or type 'exit' to quit):")
            user_prompt = input(">>> ").strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in ["exit", "quit", "q"]:
                print("\nExiting. Thank you!")
                break

            print("\nSelect Category (1: Coding, 2: Educational, 3: Research, 4: Business, 5: General):")
            cat_choice = input("Choice [Default: 5]: ").strip()
            cat_map = {"1": "Coding", "2": "Educational", "3": "Research", "4": "Business", "5": "General"}
            category = cat_map.get(cat_choice, "General")

            print("\nSelect Optimization Mode (1: Basic, 2: Advanced, 3: Expert, 4: Coding, 5: Educational):")
            mode_choice = input("Choice [Default: 2]: ").strip()
            mode_map = {"1": "Basic", "2": "Advanced", "3": "Expert", "4": "Coding", "5": "Educational"}
            mode = mode_map.get(mode_choice, "Advanced")

            print("\n[*] Processing through 6-Agent LangGraph Workflow...")
            result = run_prompt_optimization(user_prompt, category=category, mode=mode)

            dl = result.get("dl_diagnostics", {})
            comp = result.get("score_comparison", {})
            before = comp.get("before", {})
            after = comp.get("after", {})

            print("\n" + "-" * 75)
            print("DEEP LEARNING DIAGNOSTICS:")
            print(f"  • 1D-CNN Quality Class : {dl.get('cnn_prediction', {}).get('predicted_class')} ({dl.get('cnn_prediction', {}).get('confidence')}%)")
            print(f"  • Bi-LSTM Quality Class: {dl.get('lstm_prediction', {}).get('predicted_class')} ({dl.get('lstm_prediction', {}).get('confidence')}%)")
            print(f"  • Ensemble Consensus   : {dl.get('ensemble_quality')}")
            print(f"  • Overall Score Gain   : {before.get('overall')}/10  -->  {after.get('overall')}/10 ({comp.get('improvement_gain')})")
            print("-" * 75)
            print("OPTIMIZED PROMPT:")
            print(result.get("optimized_prompt", ""))
            print("-" * 75)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


def run_test_suite():
    """Runs automated verification test cases across all prompt categories and edge conditions."""
    print("\n" + "=" * 75)
    print("PHASE 10: AUTOMATED TEST SUITE EXECUTION")
    print("=" * 75)

    test_cases = [
        {"name": "Empty / Minimal Prompt", "prompt": "fix code", "cat": "Coding", "mode": "Coding"},
        {"name": "Vague Educational Prompt", "prompt": "explain machine learning", "cat": "Educational", "mode": "Educational"},
        {"name": "Research Prompt", "prompt": "summarize quantum computing", "cat": "Research", "mode": "Research"},
        {"name": "Business Prompt", "prompt": "write memo to team", "cat": "Business", "mode": "Advanced"},
        {"name": "General Travel Prompt", "prompt": "plan my trip to Japan", "cat": "General", "mode": "Basic"},
        {"name": "Detailed High-Quality Prompt", "prompt": "Act as a Senior Python Architect. Write a thread-safe ring buffer in Python with unit tests.", "cat": "Coding", "mode": "Coding"},
    ]

    passed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}/6] Running Test: {tc['name']}...")
        try:
            res = run_prompt_optimization(tc["prompt"], category=tc["cat"], mode=tc["mode"])
            assert "optimized_prompt" in res
            assert len(res["optimized_prompt"]) > 20
            assert "dl_diagnostics" in res
            print(f"    -> [PASSED] Classified as: {res['dl_diagnostics']['ensemble_quality']} | Output length: {len(res['optimized_prompt'].split())} words")
            passed += 1
        except Exception as e:
            print(f"    -> [FAILED]: {e}")

    print("\n" + "=" * 75)
    print(f"TEST SUITE SUMMARY: {passed}/{len(test_cases)} Tests Passed ({passed/len(test_cases)*100:.1f}%)")
    print("=" * 75 + "\n")


def launch_streamlit():
    """Launches the Streamlit web application."""
    app_path = os.path.join(PROJECT_ROOT, "frontend", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    print(f"[*] Launching Streamlit web application from: {app_path}")
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Prompt Optimization Assistant")
    parser.add_argument(
        "--mode",
        choices=["cli", "streamlit", "evaluate", "test"],
        default="cli",
        help="Execution mode (default: cli)",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_interactive_cli()
    elif args.mode == "streamlit":
        launch_streamlit()
    elif args.mode == "evaluate":
        run_comparative_evaluation()
    elif args.mode == "test":
        run_test_suite()
