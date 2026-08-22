"""
MediRAG Clinical Decision Support Tools.
Provides native, guideline-grounded clinical calculators and decision algorithms:
1. Blood Pressure Staging and Clinical Pathway Evaluator (WHO / ACC / AHA / ESC / NICE).
2. Renal Function & eGFR / Creatinine Clearance Calculator with Drug Dosage Adjustments.
3. Cardiovascular Disease (CVD) Risk Stratification Engine.
"""

from typing import Dict, Any, List, Optional


def classify_blood_pressure(
    systolic: int,
    diastolic: int,
    has_diabetes: bool = False,
    has_cvd: bool = False,
    has_ckd: bool = False,
) -> Dict[str, Any]:
    """
    Classifies blood pressure based on WHO (2021), ISH (2020), and ACC/AHA (2017) guidelines.
    Provides stage classification, risk tier, target BP goal, and immediate clinical recommendation.
    """
    if systolic < 70 or systolic > 300 or diastolic < 40 or diastolic > 200:
        return {
            "error": "Invalid blood pressure values. Systolic range: 70-300 mmHg, Diastolic range: 40-200 mmHg.",
            "valid": False,
        }

    # Hypertensive Crisis
    if systolic >= 180 or diastolic >= 120:
        stage = "Hypertensive Crisis (Emergency / Urgency)"
        severity = "Critical"
        recommendation = (
            "Immediate medical evaluation required. Assess for target organ damage "
            "(chest pain, shortness of breath, neurological deficits, papilledema, acute kidney injury)."
        )
        first_line_strategy = "Immediate specialist assessment and controlled BP lowering."
    elif systolic >= 140 or diastolic >= 90:
        if systolic >= 160 or diastolic >= 100:
            stage = "Stage 2 Hypertension (Grade 2)"
            severity = "High"
            recommendation = (
                "Initiate prompt pharmacological treatment with combination therapy "
                "(two first-line agents from different classes, ideally as a single-pill combination) "
                "alongside lifestyle modifications."
            )
            first_line_strategy = "Dual therapy: ACEi/ARB + CCB or Thiazide-like diuretic."
        else:
            stage = "Stage 1 Hypertension (Grade 1)"
            severity = "Moderate"
            recommendation = (
                "Initiate pharmacological treatment if cardiovascular risk is high, or if BP remains uncontrolled "
                "after 3-6 months of lifestyle intervention."
            )
            first_line_strategy = "Monotherapy or low-dose dual combination therapy."
    elif systolic >= 130 or diastolic >= 80:
        stage = "High-Normal / Prehypertension (Stage 1 ACC/AHA)"
        severity = "Mild"
        recommendation = (
            "Lifestyle interventions (DASH diet, sodium restriction < 2g/day, regular aerobic exercise, "
            "weight management, alcohol limitation). Pharmacotherapy if high CVD risk or diabetes."
        )
        first_line_strategy = "Lifestyle modification primary; pharmacotherapy if CVD risk >= 10%."
    elif systolic >= 120 and diastolic < 80:
        stage = "Elevated Blood Pressure"
        severity = "Low"
        recommendation = "Non-pharmacological lifestyle modifications and annual BP monitoring."
        first_line_strategy = "Lifestyle interventions."
    else:
        stage = "Normal / Optimal Blood Pressure"
        severity = "Optimal"
        recommendation = "Maintain healthy lifestyle habits. Re-assess blood pressure annually."
        first_line_strategy = "No pharmacological therapy indicated."

    # Determine Target Goal
    if has_diabetes or has_cvd or has_ckd:
        target_bp = "< 130/80 mmHg (High-Risk Target)"
        comorbidity_note = "Intensified BP control recommended due to presence of comorbid risk factors."
    else:
        target_bp = "< 140/90 mmHg (Standard Target; < 130/80 mmHg if tolerated)"
        comorbidity_note = "Standard guideline target goal."

    return {
        "valid": True,
        "systolic": systolic,
        "diastolic": diastolic,
        "stage": stage,
        "severity": severity,
        "target_bp": target_bp,
        "recommendation": recommendation,
        "first_line_strategy": first_line_strategy,
        "comorbidity_note": comorbidity_note,
        "guideline_sources": [
            "WHO Guideline for the pharmacological treatment of hypertension in adults (2021)",
            "NICE Clinical Guideline [NG136]: Hypertension in adults: diagnosis and management",
        ],
    }


def calculate_renal_function(
    serum_creatinine_mg_dl: float,
    age_years: int,
    is_female: bool,
    weight_kg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates estimated Glomerular Filtration Rate (eGFR) and Creatinine Clearance (CrCl).
    Provides KDIGO Chronic Kidney Disease (CKD) staging and antihypertensive dosing precautions.
    """
    if serum_creatinine_mg_dl <= 0 or age_years <= 0 or age_years > 120:
        return {
            "error": "Invalid clinical parameters. Serum creatinine must be > 0, age between 1-120.",
            "valid": False,
        }

    scr = serum_creatinine_mg_dl
    age = age_years

    # CKD-EPI 2021 (race-free standard formula)
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    gender_multiplier = 1.012 if is_female else 1.0

    scr_ratio = scr / kappa
    min_term = min(scr_ratio, 1.0) ** alpha
    max_term = max(scr_ratio, 1.0) ** -1.200
    egfr = 142 * min_term * max_term * (0.9938 ** age) * gender_multiplier
    egfr = round(egfr, 1)

    # Cockcroft-Gault CrCl (if weight provided)
    crcl = None
    if weight_kg and weight_kg > 0:
        crcl_val = ((140 - age) * weight_kg) / (72 * scr)
        if is_female:
            crcl_val *= 0.85
        crcl = round(crcl_val, 1)

    # KDIGO Staging
    if egfr >= 90:
        kdigo_stage = "G1 (Normal or high kidney function)"
        risk_level = "Low Risk"
    elif egfr >= 60:
        kdigo_stage = "G2 (Mildly decreased kidney function)"
        risk_level = "Mild Risk"
    elif egfr >= 45:
        kdigo_stage = "G3a (Mild-to-moderately decreased)"
        risk_level = "Moderate Risk"
    elif egfr >= 30:
        kdigo_stage = "G3b (Moderately-to-severely decreased)"
        risk_level = "High Risk"
    elif egfr >= 15:
        kdigo_stage = "G4 (Severely decreased kidney function)"
        risk_level = "Very High Risk"
    else:
        kdigo_stage = "G5 (Kidney failure / End-stage renal disease)"
        risk_level = "Critical / Dialysis Assessment"

    # Antihypertensive Adjustments
    precautions: List[str] = []
    if egfr < 30:
        precautions.append("Thiazide diuretics have reduced efficacy; loop diuretics (e.g. Furosemide) are preferred.")
        precautions.append("Potassium-sparing diuretics and mineralocorticoid receptor antagonists (Spironolactone) carry high hyperkalemia risk.")
        precautions.append("Monitor serum potassium and eGFR within 1-2 weeks of initiating or adjusting ACE inhibitors / ARBs.")
    elif egfr < 45:
        precautions.append("Use caution with high-dose ACE inhibitors / ARBs. Ensure regular electrolyte and creatinine monitoring.")
        precautions.append("Avoid combining ACE inhibitors with ARBs (dual blockade contraindicated).")
    else:
        precautions.append("Standard first-line antihypertensive dosing applicable.")

    return {
        "valid": True,
        "egfr_ckd_epi": egfr,
        "crcl_cockcroft_gault": crcl,
        "kdigo_stage": kdigo_stage,
        "risk_level": risk_level,
        "dosage_precautions": precautions,
        "guideline_sources": [
            "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease",
            "WHO Hypertension Guidelines - Renal Considerations",
        ],
    }


def stratify_cvd_risk(
    age: int,
    systolic_bp: int,
    is_smoker: bool = False,
    has_diabetes: bool = False,
    total_cholesterol_mg_dl: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Simplified guideline-grounded Cardiovascular Risk Stratification.
    """
    points = 0
    if age >= 65:
        points += 3
    elif age >= 50:
        points += 2
    elif age >= 40:
        points += 1

    if systolic_bp >= 160:
        points += 3
    elif systolic_bp >= 140:
        points += 2
    elif systolic_bp >= 130:
        points += 1

    if is_smoker:
        points += 2
    if has_diabetes:
        points += 3

    if total_cholesterol_mg_dl and total_cholesterol_mg_dl >= 240:
        points += 2
    elif total_cholesterol_mg_dl and total_cholesterol_mg_dl >= 200:
        points += 1

    if points >= 7 or has_diabetes or (systolic_bp >= 160 and age >= 50):
        tier = "High CVD Risk (> 20% 10-year risk)"
        action = "Immediate pharmacological therapy and aggressive risk factor reduction."
    elif points >= 4:
        tier = "Moderate CVD Risk (10-20% 10-year risk)"
        action = "Lifestyle intervention with early pharmacological initiation if BP remains above target."
    else:
        tier = "Low CVD Risk (< 10% 10-year risk)"
        action = "Lifestyle modifications with periodic re-evaluation."

    return {
        "valid": True,
        "risk_points": points,
        "risk_tier": tier,
        "clinical_action": action,
        "target_bp": "< 130/80 mmHg" if points >= 4 else "< 140/90 mmHg",
    }
