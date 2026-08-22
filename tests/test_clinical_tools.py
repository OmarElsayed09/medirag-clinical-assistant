"""
Unit tests for MediRAG Clinical Decision Support Tools (src/clinical_tools.py).
Tests:
- Blood Pressure Staging according to WHO and ACC/AHA guidelines
- Renal Function: CKD-EPI 2021 eGFR and Cockcroft-Gault CrCl calculations
- Cardiovascular Risk Stratification
"""

import pytest
from src.clinical_tools import (
    classify_blood_pressure,
    calculate_renal_function,
    stratify_cvd_risk,
)


class TestBloodPressureClassification:
    def test_optimal_normal_bp(self):
        result = classify_blood_pressure(systolic=115, diastolic=75)
        assert result["valid"] is True
        assert "Normal" in result["stage"]
        assert result["severity"] == "Optimal"
        assert result["target_bp"].startswith("< 140/90")

    def test_stage_1_hypertension(self):
        result = classify_blood_pressure(systolic=145, diastolic=92)
        assert result["valid"] is True
        assert "Stage 1" in result["stage"]
        assert result["severity"] == "Moderate"

    def test_stage_2_hypertension(self):
        result = classify_blood_pressure(systolic=165, diastolic=102)
        assert result["valid"] is True
        assert "Stage 2" in result["stage"]
        assert result["severity"] == "High"
        assert "Dual therapy" in result["first_line_strategy"]

    def test_hypertensive_crisis(self):
        result = classify_blood_pressure(systolic=190, diastolic=125)
        assert result["valid"] is True
        assert "Crisis" in result["stage"]
        assert result["severity"] == "Critical"

    def test_comorbid_target_bp(self):
        result = classify_blood_pressure(systolic=135, diastolic=85, has_diabetes=True)
        assert result["valid"] is True
        assert "< 130/80" in result["target_bp"]

    def test_invalid_bp_ranges(self):
        result = classify_blood_pressure(systolic=50, diastolic=30)
        assert result["valid"] is False
        assert "error" in result


class TestRenalFunctionCalculator:
    def test_ckd_epi_normal_male(self):
        # 45yo male, SCr 0.9 mg/dL
        result = calculate_renal_function(serum_creatinine_mg_dl=0.9, age_years=45, is_female=False)
        assert result["valid"] is True
        assert result["egfr_ckd_epi"] >= 90
        assert "G1" in result["kdigo_stage"]

    def test_ckd_epi_impaired_female(self):
        # 70yo female, SCr 2.2 mg/dL
        result = calculate_renal_function(serum_creatinine_mg_dl=2.2, age_years=70, is_female=True)
        assert result["valid"] is True
        assert result["egfr_ckd_epi"] < 30
        assert any("loop diuretics" in p.lower() for p in result["dosage_precautions"])

    def test_cockcroft_gault_with_weight(self):
        # 60yo male, 80kg, SCr 1.0 mg/dL -> CrCl = ((140-60)*80)/(72*1.0) = 88.9
        result = calculate_renal_function(serum_creatinine_mg_dl=1.0, age_years=60, is_female=False, weight_kg=80.0)
        assert result["valid"] is True
        assert result["crcl_cockcroft_gault"] is not None
        assert abs(result["crcl_cockcroft_gault"] - 88.9) < 0.5

    def test_invalid_renal_parameters(self):
        result = calculate_renal_function(serum_creatinine_mg_dl=-1.0, age_years=50, is_female=False)
        assert result["valid"] is False
        assert "error" in result


class TestCVDRiskStratification:
    def test_high_risk_diabetic_patient(self):
        result = stratify_cvd_risk(age=62, systolic_bp=155, is_smoker=True, has_diabetes=True)
        assert result["valid"] is True
        assert "High" in result["risk_tier"]
        assert "< 130/80" in result["target_bp"]

    def test_low_risk_young_patient(self):
        result = stratify_cvd_risk(age=30, systolic_bp=125, is_smoker=False, has_diabetes=False)
        assert result["valid"] is True
        assert "Low" in result["risk_tier"]
