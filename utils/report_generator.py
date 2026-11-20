"""
==============================================================================
FILE: utils/report_generator.py
PURPOSE:
    Generates professional PDF and Markdown optimization reports for the
    Intelligent Prompt Optimization Assistant.
    Summarizes before-and-after scores, DL predictions (1D-CNN vs Bi-LSTM),
    detected weaknesses, and the final optimized prompt.
==============================================================================
"""

import os
from datetime import datetime
from typing import Any, Dict


def generate_text_report(data: Dict[str, Any]) -> str:
    """Generates a clean, formatted text audit report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dl = data.get("dl_diagnostics", {})
    comp = data.get("score_comparison", {})
    before = comp.get("before", {})
    after = comp.get("after", {})

    report = f"""================================================================================
INTELLIGENT PROMPT OPTIMIZATION ASSISTANT - AUDIT REPORT
Generated: {timestamp}
================================================================================

[1] PROMPT METADATA:
--------------------------------------------------------------------------------
Domain / Category   : {data.get('category', 'General')}
Optimization Mode   : {data.get('optimization_mode', 'Advanced')}
Review Status       : {data.get('review_summary', {}).get('status', 'APPROVED')}

[2] DEEP LEARNING MODEL DIAGNOSTICS:
--------------------------------------------------------------------------------
1D-CNN Prediction   : {dl.get('cnn_prediction', {}).get('predicted_class')} (Confidence: {dl.get('cnn_prediction', {}).get('confidence')}%)
Bi-LSTM Prediction  : {dl.get('lstm_prediction', {}).get('predicted_class')} (Confidence: {dl.get('lstm_prediction', {}).get('confidence')}%)
Ensemble Consensus  : {dl.get('ensemble_quality', 'Average')}

[3] QUALITY SCORES (BEFORE vs AFTER OPTIMIZATION):
--------------------------------------------------------------------------------
Metric              | Before (Original)  | After (Optimized)  | Improvement
--------------------+--------------------+--------------------+-------------
Overall Quality     | {before.get('overall', 0):>14.1f}/10 | {after.get('overall', 0):>14.1f}/10 | {comp.get('improvement_gain', 'N/A')}
Clarity Score       | {before.get('clarity', 0):>14.1f}/10 | {after.get('clarity', 0):>14.1f}/10 | Resolved Ambiguity
Context Score       | {before.get('context', 0):>14.1f}/10 | {after.get('context', 0):>14.1f}/10 | Injected Domain Context
Specificity Score   | {before.get('specificity', 0):>14.1f}/10 | {after.get('specificity', 0):>14.1f}/10 | Precise Parameters
Completeness Score  | {before.get('completeness', 0):>14.1f}/10 | {after.get('completeness', 0):>14.1f}/10 | Structured Deliverable

[4] DETECTED WEAKNESSES IN ORIGINAL PROMPT:
--------------------------------------------------------------------------------
"""
    for w in data.get("detected_weaknesses", []):
        report += f"  * {w}\n"

    report += f"""
[5] ENHANCEMENTS APPLIED:
--------------------------------------------------------------------------------
"""
    for imp in data.get("improvements_applied", []):
        report += f"  * {imp}\n"

    report += f"""
[6] ORIGINAL USER PROMPT:
--------------------------------------------------------------------------------
{data.get('original_prompt', '')}

[7] FINAL OPTIMIZED PRODUCTION-READY PROMPT:
--------------------------------------------------------------------------------
{data.get('optimized_prompt', '')}
================================================================================
"""
    return report


def generate_pdf_report(data: Dict[str, Any], output_path: str) -> bool:
    """
    Generates a PDF audit report using fpdf2 if installed,
    or falls back to a clean text report.
    """
    try:
        from fpdf import FPDF

        class PDFReport(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 14)
                self.cell(0, 10, "Intelligent Prompt Optimization Assistant", ln=True, align="C")
                self.set_font("Helvetica", "I", 9)
                self.cell(0, 5, "Major Project: CNN + LSTM + Agentic AI Architecture", ln=True, align="C")
                self.line(10, 26, 200, 26)
                self.ln(6)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        pdf = PDFReport()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # 1. Title & Metadata
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Prompt Optimization Audit & Diagnostic Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Category / Domain: {data.get('category')}  |  Mode: {data.get('optimization_mode')}", ln=True)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(4)

        # 2. Deep Learning Scores Table
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "1. Deep Learning Model Evaluations", ln=True)
        pdf.set_font("Helvetica", "", 9)
        dl = data.get("dl_diagnostics", {})
        pdf.cell(0, 5, f"1D-CNN Prediction: {dl.get('cnn_prediction', {}).get('predicted_class')} (Confidence: {dl.get('cnn_prediction', {}).get('confidence')}%)", ln=True)
        pdf.cell(0, 5, f"Bi-LSTM Prediction: {dl.get('lstm_prediction', {}).get('predicted_class')} (Confidence: {dl.get('lstm_prediction', {}).get('confidence')}%)", ln=True)
        pdf.cell(0, 5, f"Ensemble Quality Consensus: {dl.get('ensemble_quality')}", ln=True)
        pdf.ln(4)

        # 3. Before vs After Scores
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "2. Quality Metrics (Before vs After)", ln=True)
        pdf.set_font("Helvetica", "", 9)
        comp = data.get("score_comparison", {})
        before = comp.get("before", {})
        after = comp.get("after", {})
        pdf.cell(0, 5, f"Overall Quality Score: {before.get('overall')}/10  -->  {after.get('overall')}/10  ({comp.get('improvement_gain')})", ln=True)
        pdf.cell(0, 5, f"Clarity: {before.get('clarity')}/10 -> {after.get('clarity')}/10  |  Context: {before.get('context')}/10 -> {after.get('context')}/10", ln=True)
        pdf.cell(0, 5, f"Specificity: {before.get('specificity')}/10 -> {after.get('specificity')}/10  |  Completeness: {before.get('completeness')}/10 -> {after.get('completeness')}/10", ln=True)
        pdf.ln(4)

        # 4. Original Prompt
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "3. Original User Prompt", ln=True)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, data.get("original_prompt", ""))
        pdf.ln(4)

        # 5. Final Optimized Prompt
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "4. Production-Ready Optimized Prompt", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, data.get("optimized_prompt", ""))

        pdf.output(output_path)
        return True
    except Exception as e:
        # Fallback to text file
        txt_path = output_path.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(generate_text_report(data))
        return False
