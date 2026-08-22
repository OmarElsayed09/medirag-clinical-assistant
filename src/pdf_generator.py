"""
MediRAG Medical-Grade PDF Export Engine.
Generates structured Clinical SOAP Decision Support Reports and Patient Discharge Leaflets.
"""

import io
from datetime import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running footer with timestamp and page numbers to every page."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Top Header Rule
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 756, 576, 756)

        # Running Bottom Footer Rule
        self.line(36, 40, 576, 40)

        # Footer Text
        footer_text = "MediRAG Clinical Decision Support System | Strictly Grounded in WHO & NICE Guidelines"
        self.drawString(36, 28, footer_text)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 28, page_str)
        self.restoreState()


def _get_clinical_styles():
    """Builds a cohesive palette and typography set for clinical documents."""
    base_styles = getSampleStyleSheet()
    
    styles = {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=3,
        ),
        "subtitle": ParagraphStyle(
            "DocSubTitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
            spaceAfter=8,
        ),
        "section_header": ParagraphStyle(
            "SectionHeader",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#0284c7"),
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "DocBody",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#1e293b"),
        ),
        "body_bold": ParagraphStyle(
            "DocBodyBold",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#0f172a"),
        ),
        "alert_title": ParagraphStyle(
            "AlertTitle",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#991b1b"),
        ),
        "alert_body": ParagraphStyle(
            "AlertBody",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#7f1d1d"),
        ),
        "guideline_tag": ParagraphStyle(
            "GuidelineTag",
            parent=base_styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#0369a1"),
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#1e293b"),
        ),
        "table_cell_bold": ParagraphStyle(
            "TableCellBold",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#0f172a"),
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
        ),
        "disclaimer": ParagraphStyle(
            "DisclaimerText",
            parent=base_styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#64748b"),
        ),
    }
    return styles


def generate_clinical_soap_pdf(
    case_result: Dict[str, Any],
    clinician_name: Optional[str] = "Clinical Staff",
    institution_name: Optional[str] = "MediRAG Clinical Decision Support",
) -> bytes:
    """
    Renders an evidence-based, hospital-grade SOAP Clinical Report in PDF format.
    
    Args:
        case_result: Dictionary returned from analyze_patient_case(...)
        clinician_name: Optional name of the evaluating healthcare provider
        institution_name: Medical department or hospital name
        
    Returns:
        Binary bytes of the generated PDF document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=46,
    )
    
    styles = _get_clinical_styles()
    story: List[Any] = []
    
    # 1. Header Banner
    story.append(Paragraph("MediRAG | Clinical Decision Support Report", styles["title"]))
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    subtitle_text = f"Evaluated by: <b>{clinician_name}</b> | Facility: <b>{institution_name}</b> | Generated: {timestamp_str}"
    story.append(Paragraph(subtitle_text, styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # 2. Patient Summary & Risk Metrics Grid
    summary = case_result.get("patient_summary", {})
    age = summary.get("age", "N/A")
    sex = summary.get("sex", "N/A")
    bp = summary.get("bp", "N/A")
    bp_stage = summary.get("bp_stage", "N/A")
    severity = summary.get("severity_tier", "Standard")
    target_bp = summary.get("target_bp", "< 140/90 mmHg")
    cvd_risk = summary.get("cvd_risk_tier", "Not Stratified")
    renal_stage = summary.get("renal_stage", "Pending")
    egfr_val = summary.get("egfr", "N/A")

    patient_grid_data = [
        [
            Paragraph("<b>Patient Age:</b>", styles["table_cell_bold"]),
            Paragraph(f"{age} years", styles["table_cell"]),
            Paragraph("<b>Blood Pressure:</b>", styles["table_cell_bold"]),
            Paragraph(f"{bp}", styles["table_cell"]),
        ],
        [
            Paragraph("<b>Biological Sex:</b>", styles["table_cell_bold"]),
            Paragraph(f"{sex}", styles["table_cell"]),
            Paragraph("<b>Clinical BP Staging:</b>", styles["table_cell_bold"]),
            Paragraph(f"<b>{bp_stage}</b> ({severity})", styles["table_cell"]),
        ],
        [
            Paragraph("<b>10-Yr CVD Risk Tier:</b>", styles["table_cell_bold"]),
            Paragraph(f"<b>{cvd_risk}</b>", styles["table_cell"]),
            Paragraph("<b>Target Blood Pressure:</b>", styles["table_cell_bold"]),
            Paragraph(f"<b>{target_bp}</b>", styles["table_cell"]),
        ],
        [
            Paragraph("<b>KDIGO Renal Stage:</b>", styles["table_cell_bold"]),
            Paragraph(f"{renal_stage}", styles["table_cell"]),
            Paragraph("<b>CKD-EPI eGFR:</b>", styles["table_cell_bold"]),
            Paragraph(f"{egfr_val} mL/min/1.73m²" if egfr_val else "Not provided", styles["table_cell"]),
        ],
    ]

    grid_table = Table(patient_grid_data, colWidths=[110, 155, 125, 150])
    grid_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(grid_table)
    story.append(Spacer(1, 10))

    # 3. Drug-Drug Interaction (DDI) Alert Banner (if applicable)
    ddi_data = case_result.get("drug_interactions")
    if ddi_data and ddi_data.get("interactions"):
        story.append(Paragraph("CRITICAL DRUG-DRUG INTERACTIONS (DDI) ALERT", styles["section_header"]))
        for inter in ddi_data["interactions"]:
            is_critical = inter.get("severity_code") == "critical"
            bg_color = colors.HexColor("#fef2f2") if is_critical else colors.HexColor("#fffbeb")
            border_color = colors.HexColor("#ef4444") if is_critical else colors.HexColor("#f59e0b")

            alert_content = [
                [Paragraph(f"<b>{inter.get('title', 'Interaction Alert')}</b> ({inter.get('severity')})", styles["alert_title"])],
                [Paragraph(f"<b>Drugs Involved:</b> {', '.join(inter.get('drugs_involved', []))}", styles["alert_body"])],
                [Paragraph(f"<b>Mechanism:</b> {inter.get('mechanism', '')}", styles["alert_body"])],
                [Paragraph(f"<b>Recommended Clinical Action:</b> {inter.get('clinical_action', '')}", styles["alert_body"])],
            ]
            alert_table = Table(alert_content, colWidths=[540])
            alert_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                ("BOX", (0, 0), (-1, -1), 1.2, border_color),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(alert_table)
            story.append(Spacer(1, 6))
    else:
        # Safe Badge
        safe_table = Table(
            [[Paragraph("<b>Medication Safety Audit:</b> No critical or contraindicated drug interactions detected in active regimen.", styles["table_cell"])]],
            colWidths=[540],
        )
        safe_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#86efac")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(safe_table)
        story.append(Spacer(1, 8))

    # 4. Structured SOAP Clinical Note
    soap = case_result.get("soap_note", {})
    story.append(Paragraph("STRUCTURED CLINICAL SOAP NOTE", styles["section_header"]))

    soap_table_data = [
        [Paragraph("<b>S (Subjective)</b>", styles["table_cell_bold"]), Paragraph(soap.get("subjective", "N/A"), styles["table_cell"])],
        [Paragraph("<b>O (Objective)</b>", styles["table_cell_bold"]), Paragraph(soap.get("objective", "N/A"), styles["table_cell"])],
        [Paragraph("<b>A (Assessment)</b>", styles["table_cell_bold"]), Paragraph(soap.get("assessment", "N/A"), styles["table_cell"])],
    ]
    soap_table = Table(soap_table_data, colWidths=[100, 440])
    soap_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#ffffff")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(soap_table)
    story.append(Spacer(1, 10))

    # 5. Clinical Plan & Stepwise Guideline Pathway (P of SOAP)
    story.append(Paragraph("<b>P (Plan) - Stepwise Clinical Pathway & Guideline Evidence:</b>", styles["body_bold"]))
    story.append(Spacer(1, 4))

    pathway = case_result.get("clinical_pathway", [])
    if pathway:
        pathway_rows = [
            [
                Paragraph("<b>Step</b>", styles["table_header"]),
                Paragraph("<b>Clinical Action & Strategy</b>", styles["table_header"]),
                Paragraph("<b>Guideline Evidence Source</b>", styles["table_header"]),
            ]
        ]
        for step_item in pathway:
            step_num = str(step_item.get("step", "•"))
            step_title = step_item.get("title", "")
            step_desc = step_item.get("description", "")
            step_guide = step_item.get("guideline", "Official Medical Guideline")

            action_p = Paragraph(f"<b>{step_title}</b><br/>{step_desc}", styles["table_cell"])
            guide_p = Paragraph(f"{step_guide}", styles["guideline_tag"])

            pathway_rows.append([
                Paragraph(f"<b>{step_num}</b>", styles["table_cell_bold"]),
                action_p,
                guide_p,
            ])

        pathway_table = Table(pathway_rows, colWidths=[36, 324, 180])
        pathway_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(pathway_table)
    else:
        story.append(Paragraph("Routine monitoring and lifestyle modifications indicated.", styles["body"]))

    story.append(Spacer(1, 14))

    # 6. Disclaimer & Verification Notice
    disclaimer_text = (
        "<b>Medical Legal Disclaimer:</b> This report was generated by MediRAG for clinical reference "
        "and decision support, grounded exclusively in official WHO and NICE hypertension guidelines. "
        "It does not replace individual clinical judgment, diagnostic verification, or formal medical consultation."
    )
    story.append(KeepTogether([
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=5),
        Paragraph(disclaimer_text, styles["disclaimer"]),
    ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


def generate_patient_leaflet_pdf(guide_data: Dict[str, Any]) -> bytes:
    """
    Renders a patient-friendly discharge education guide in PDF format.
    
    Args:
        guide_data: Dictionary returned from generate_patient_education_guide(...)
        
    Returns:
        Binary bytes of the generated PDF document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=46,
    )
    
    styles = _get_clinical_styles()
    story: List[Any] = []

    # Title Banner
    title_text = guide_data.get("title", "Patient Health & Medication Guide")
    story.append(Paragraph(title_text, styles["title"]))
    timestamp_str = datetime.now().strftime("%Y-%m-%d")
    story.append(Paragraph(f"Patient Discharge & Self-Care Instructions | Issued on {timestamp_str}", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#059669"), spaceAfter=10))

    # Condition & Target Summary
    summary_text = guide_data.get("condition_summary", "")
    if summary_text:
        summary_table = Table(
            [[Paragraph(f"<b>Treatment Objective & Summary:</b><br/>{summary_text}", styles["table_cell"])]],
            colWidths=[540],
        )
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ecfdf5")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#a7f3d0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 10))

    # Section 1: Lifestyle & Nutrition Rules
    lifestyle = guide_data.get("lifestyle_rules", [])
    if lifestyle:
        story.append(Paragraph("1. Lifestyle & Nutrition Recommendations", styles["section_header"]))
        for item in lifestyle:
            story.append(Paragraph(f"• {item}", styles["body"]))
            story.append(Spacer(1, 2))
        story.append(Spacer(1, 8))

    # Section 2: Medication Safety Tips
    med_tips = guide_data.get("medication_safety_tips", [])
    if med_tips:
        story.append(Paragraph("2. Medication Safety & Daily Adherence", styles["section_header"]))
        for tip in med_tips:
            story.append(Paragraph(f"• {tip}", styles["body"]))
            story.append(Spacer(1, 2))
        story.append(Spacer(1, 8))

    # Section 3: Emergency Red Flags (Acute Warning Signs)
    red_flags = guide_data.get("emergency_red_flags", [])
    if red_flags:
        story.append(Paragraph("3. Warning Signs Requiring Immediate Emergency Attention (Red Flags)", styles["section_header"]))
        red_flag_items = "<br/>".join([f"• <b>{flag}</b>" for flag in red_flags])
        action_text = guide_data.get("emergency_action", "Call Emergency Services Immediately.")

        emergency_table = Table(
            [[
                Paragraph(
                    f"{red_flag_items}<br/><br/><b>Action Required:</b> {action_text}",
                    styles["alert_body"],
                )
            ]],
            colWidths=[540],
        )
        emergency_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
            ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#ef4444")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(emergency_table)
        story.append(Spacer(1, 10))

    # Legal Disclaimer
    disclaimer_text = (
        "<b>Notice:</b> This patient education material is provided for personal health guidance "
        "and is based on recognized clinical guidelines. Contact your attending physician or "
        "local emergency clinic if you experience any worsening symptoms."
    )
    story.append(KeepTogether([
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=5),
        Paragraph(disclaimer_text, styles["disclaimer"]),
    ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
