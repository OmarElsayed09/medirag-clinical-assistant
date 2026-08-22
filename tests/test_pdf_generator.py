"""
Unit tests for the MediRAG PDF Export Engine.
"""

from src.pdf_generator import generate_clinical_soap_pdf, generate_patient_leaflet_pdf
from src.clinical_expert import analyze_patient_case, generate_patient_education_guide


def test_generate_clinical_soap_pdf():
    # Generate full case analysis
    case_result = analyze_patient_case(
        age=62,
        is_female=False,
        systolic_bp=165,
        diastolic_bp=102,
        has_diabetes=True,
        has_cvd=False,
        has_ckd=False,
        is_smoker=True,
        serum_creatinine_mg_dl=1.4,
        weight_kg=80.0,
        current_medications=["ace_inhibitor", "spironolactone"],
    )

    pdf_bytes = generate_clinical_soap_pdf(
        case_result=case_result,
        clinician_name="Dr. Smith, MD",
        institution_name="Cardiology Clinical Unit",
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # Check PDF Magic Header
    assert pdf_bytes[:5] == b"%PDF-"


def test_generate_clinical_soap_pdf_without_ddi():
    case_result = analyze_patient_case(
        age=45,
        is_female=True,
        systolic_bp=135,
        diastolic_bp=85,
        current_medications=["thiazide_diuretic"],
    )

    pdf_bytes = generate_clinical_soap_pdf(case_result=case_result)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes[:5] == b"%PDF-"


def test_generate_patient_leaflet_pdf():
    guide_data = generate_patient_education_guide(
        condition_name="Stage 2 Essential Hypertension",
        target_bp="< 130/80 mmHg",
        language="English",
    )

    pdf_bytes = generate_patient_leaflet_pdf(guide_data=guide_data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes[:5] == b"%PDF-"
