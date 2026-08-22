"""
MediRAG Advanced Clinical Expert Engine.
Implements high-value clinical features:
1. Patient Case Analyzer & SOAP Clinical Pathway Generator.
2. Drug-Drug Interaction (DDI) & Contraindication Matrix.
3. Patient-Friendly Health & Discharge Education Leaflet Generator.
4. Multi-Guideline Consensus & Divergence Matrix (WHO vs NICE vs ACC/AHA).
"""

from typing import Dict, Any, List, Optional
from src.clinical_tools import classify_blood_pressure, calculate_renal_function, stratify_cvd_risk


# --- DRUG INTERACTION CLINICAL DATABASE ---
DDI_DATABASE = [
    {
        "pair": frozenset(["ace_inhibitor", "arb"]),
        "drugs": ["Lisinopril/Enalapril/Ramipril", "Losartan/Valsartan/Candesartan"],
        "severity": "Severe (Contraindicated)",
        "severity_code": "critical",
        "title": "Dual Renin-Angiotensin System (RAS) Blockade",
        "mechanism": "Combining ACE inhibitors with ARBs does not improve cardiovascular outcomes and significantly increases the risk of acute renal failure, severe hypotension, and hyperkalemia.",
        "action": "Avoid combination. Choose either an ACE inhibitor OR an ARB, not both (WHO & NICE guidelines contraindicated).",
    },
    {
        "pair": frozenset(["ace_inhibitor", "spironolactone"]),
        "drugs": ["ACE Inhibitors", "Spironolactone / Eplerenone (MRA)"],
        "severity": "Moderate (High-Risk Caution)",
        "severity_code": "high",
        "title": "Potassium-Sparing Hyperkalemia Risk",
        "mechanism": "Both drug classes decrease renal potassium excretion, substantially increasing the risk of life-threatening hyperkalemia, especially with impaired eGFR.",
        "action": "Monitor serum creatinine, eGFR, and serum potassium at 1 week, 4 weeks, and every 3-6 months. Caution if baseline potassium > 5.0 mmol/L or eGFR < 30.",
    },
    {
        "pair": frozenset(["arb", "spironolactone"]),
        "drugs": ["ARBs", "Spironolactone / Eplerenone (MRA)"],
        "severity": "Moderate (High-Risk Caution)",
        "severity_code": "high",
        "title": "Potassium-Sparing Hyperkalemia Risk",
        "mechanism": "Combined inhibition of aldosterone leads to synergistic potassium retention.",
        "action": "Mandatory electrolyte and renal function surveillance. Avoid potassium supplements and salt substitutes containing potassium.",
    },
    {
        "pair": frozenset(["beta_blocker", "verapamil_diltiazem"]),
        "drugs": ["Beta Blockers (e.g. Bisoprolol, Metoprolol)", "Non-DHP CCB (Verapamil, Diltiazem)"],
        "severity": "Severe (Caution / Avoid)",
        "severity_code": "critical",
        "title": "Additive Bradycardia and AV Conduction Block",
        "mechanism": "Both classes exert negative inotropic and negative dromotropic effects on the sinoatrial and atrioventricular nodes, risking severe bradycardia, AV block, and heart failure exacerbation.",
        "action": "Avoid concomitant use unless under specialist cardiology supervision. Dihydropyridine CCBs (Amlodipine) are the safe alternative.",
    },
    {
        "pair": frozenset(["ace_inhibitor", "nsaid"]),
        "drugs": ["ACE Inhibitors", "NSAIDs (Ibuprofen, Naproxen, Diclofenac)"],
        "severity": "Moderate (Renal Risk)",
        "severity_code": "moderate",
        "title": "Hemodynamic Renal Impairment & BP Blunting",
        "mechanism": "NSAIDs inhibit vasodilatory prostaglandins at the afferent arteriole, while ACE inhibitors dilate the efferent arteriole, causing acute drop in glomerular filtration pressure and blunting BP lowering.",
        "action": "Avoid chronic NSAID use in hypertensive patients. Use Paracetamol for analgesia; monitor renal function and BP.",
    },
    {
        "pair": frozenset(["arb", "nsaid"]),
        "drugs": ["ARBs", "NSAIDs (Ibuprofen, Naproxen, Diclofenac)"],
        "severity": "Moderate (Renal Risk)",
        "severity_code": "moderate",
        "title": "Hemodynamic Renal Impairment & BP Blunting",
        "mechanism": "Synergistic reduction in renal blood flow and loss of blood pressure control.",
        "action": "Minimize NSAID duration. Check creatinine and BP if combined.",
    },
    {
        "pair": frozenset(["thiazide_diuretic", "nsaid"]),
        "drugs": ["Thiazide / Thiazide-like Diuretics (Chlorthalidone, Indapamide)", "NSAIDs"],
        "severity": "Mild to Moderate",
        "severity_code": "moderate",
        "title": "Attenuated Diuretic and Antihypertensive Efficacy",
        "mechanism": "Sodium and water retention induced by NSAIDs antagonizes diuretic mechanism.",
        "action": "Monitor blood pressure; consider non-NSAID alternatives.",
    },
]


def check_drug_interactions(drugs_selected: List[str]) -> Dict[str, Any]:
    """
    Evaluates a list of medication class identifiers for dangerous clinical interactions.
    """
    found_interactions = []
    selected_set = set(d.strip().lower() for d in drugs_selected)

    for rule in DDI_DATABASE:
        if rule["pair"].issubset(selected_set):
            found_interactions.append({
                "title": rule["title"],
                "severity": rule["severity"],
                "severity_code": rule["severity_code"],
                "drugs_involved": rule["drugs"],
                "mechanism": rule["mechanism"],
                "clinical_action": rule["action"],
            })

    return {
        "total_drugs_checked": len(drugs_selected),
        "total_interactions_found": len(found_interactions),
        "has_critical": any(i["severity_code"] == "critical" for i in found_interactions),
        "interactions": found_interactions,
    }


def analyze_patient_case(
    age: int,
    is_female: bool,
    systolic_bp: int,
    diastolic_bp: int,
    has_diabetes: bool = False,
    has_cvd: bool = False,
    has_ckd: bool = False,
    is_smoker: bool = False,
    serum_creatinine_mg_dl: Optional[float] = None,
    weight_kg: Optional[float] = None,
    current_medications: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Synthesizes a full clinical SOAP note and customized clinical pathway.
    """
    # 1. Classify Blood Pressure
    bp_eval = classify_blood_pressure(systolic_bp, diastolic_bp, has_diabetes, has_cvd, has_ckd)

    # 2. Renal Evaluation
    renal_eval = None
    if serum_creatinine_mg_dl and serum_creatinine_mg_dl > 0:
        renal_eval = calculate_renal_function(serum_creatinine_mg_dl, age, is_female, weight_kg)

    # 3. CVD Risk Stratification
    cvd_eval = stratify_cvd_risk(age, systolic_bp, is_smoker, has_diabetes)

    # 4. Drug Interaction Check on current medications
    ddi_eval = None
    if current_medications:
        ddi_eval = check_drug_interactions(current_medications)

    # 5. Build Synthesized Clinical Recommendations Pathway
    pathway_steps = []
    
    # Step 1: BP Initiation / Strategy
    if systolic_bp >= 160 or diastolic_bp >= 100:
        pathway_steps.append({
            "step": 1,
            "title": "Immediate Dual Combination Initiation",
            "description": "Initiate prompt two-drug pharmacological treatment with a single-pill combination (SPC) combining an ACEi/ARB + CCB or Thiazide-like diuretic.",
            "guideline": "WHO 2021 Recommendation 3 (Strong recommendation, moderate-certainty evidence)"
        })
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        pathway_steps.append({
            "step": 1,
            "title": "First-Line Pharmacological Initiation",
            "description": "Initiate first-line monotherapy or low-dose dual combination alongside structured lifestyle interventions.",
            "guideline": "WHO 2021 & NICE NG136 Recommendation"
        })
    else:
        pathway_steps.append({
            "step": 1,
            "title": "Lifestyle Modification & Monitoring",
            "description": "Reinforce non-pharmacological interventions: dietary sodium < 2g/day, DASH dietary pattern, regular physical activity, weight management.",
            "guideline": "WHO 2021 Lifestyle Recommendations"
        })

    # Step 2: Target Blood Pressure Goal
    target_text = bp_eval.get("target_bp", "< 140/90 mmHg")
    pathway_steps.append({
        "step": 2,
        "title": "Target Blood Pressure Goal",
        "description": f"Target blood pressure goal is {target_text}. Re-evaluate within 4 weeks of initiating or intensifying therapy.",
        "guideline": "WHO 2021 Recommendation 5"
    })

    # Step 3: Specific Comorbidity & Renal Adjustments
    if renal_eval and renal_eval.get("egfr_ckd_epi", 100) < 45:
        pathway_steps.append({
            "step": 3,
            "title": "Renal Function Monitoring & Diuretic Selection",
            "description": f"eGFR is {renal_eval.get('egfr_ckd_epi')} mL/min/1.73m2 ({renal_eval.get('kdigo_stage')}). Use caution with potassium-sparing agents; check serum potassium and creatinine at 2 weeks.",
            "guideline": "KDIGO 2024 & WHO Guideline Renal Guidance"
        })

    # Step 4: CVD Risk Mitigation
    if cvd_eval.get("risk_points", 0) >= 4 or has_diabetes:
        pathway_steps.append({
            "step": 4,
            "title": "Comprehensive Cardiovascular Risk Reduction",
            "description": "Evaluate lipid profile (Statin therapy indicated if CVD risk > 10% or diabetes). Encourage smoking cessation.",
            "guideline": "WHO CVD Prevention Guidelines"
        })

    # Assemble Structured SOAP Note
    soap_note = {
        "subjective": f"Patient is a {age}-year-old {'female' if is_female else 'male'} presenting for clinical evaluation. Tobacco smoker: {'Yes' if is_smoker else 'No'}. Relevant history: {'Diabetes Mellitus, ' if has_diabetes else ''}{'Cardiovascular Disease, ' if has_cvd else ''}{'Chronic Kidney Disease' if has_ckd else 'None'}.",
        "objective": f"Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg. " + (f"Serum Creatinine: {serum_creatinine_mg_dl} mg/dL (eGFR: {renal_eval.get('egfr_ckd_epi')} mL/min/1.73m2, {renal_eval.get('kdigo_stage')})." if renal_eval else "Labs: Pending."),
        "assessment": f"Primary Assessment: {bp_eval.get('stage')} ({bp_eval.get('severity')} Tier). Cardiovascular Risk: {cvd_eval.get('risk_tier')}.",
        "plan": pathway_steps,
    }

    return {
        "patient_summary": {
            "age": age,
            "sex": "Female" if is_female else "Male",
            "bp": f"{systolic_bp}/{diastolic_bp} mmHg",
            "bp_stage": bp_eval.get("stage"),
            "severity_tier": bp_eval.get("severity"),
            "target_bp": bp_eval.get("target_bp"),
            "cvd_risk_tier": cvd_eval.get("risk_tier"),
            "renal_stage": renal_eval.get("kdigo_stage") if renal_eval else "Not Provided",
            "egfr": renal_eval.get("egfr_ckd_epi") if renal_eval else None,
        },
        "soap_note": soap_note,
        "clinical_pathway": pathway_steps,
        "drug_interactions": ddi_eval,
    }


def generate_patient_education_guide(
    condition_name: str,
    target_bp: str,
    language: str = "Arabic",
    medications: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generates plain-language, patient-friendly health education and medication guides.
    """
    if language.lower() == "arabic":
        title = "دليل المريض الإرشادي لصحة ضغط الدم والقلب"
        summary = (
            f"تم تشخيص حالتك بـ {condition_name}. الهدف العلاجي لضغط الدم هو الوصول إلى أقل من {target_bp}. "
            "الالتزام بنمط الحياة الصحي وتناول الأدوية بانتظام يقلل بشكل كبير من مخاطر الأزمات القلبية والسكتات الدماغية."
        )
        lifestyle_rules = [
            "تقليل ملح الطعام في الوجبات إلى أقل من ملعقة صغيرة يومياً (أقل من 2 جرام صوديوم).",
            "اتباع نمط غذائي غني بالخضراوات، الفواكه، والحبوب الكاملة (حمية DASH).",
            "ممارسة النشاط البدني المعتدل (مثل المشي السريع) لمدة 30 دقيقة، 5 أيام أسبوعياً.",
            "الامتناع التام عن التدخين بجميع أنواعه.",
            "الحفاظ على وزن صحي وتجنب الإجهاد والتوتر العصبي.",
        ]
        medication_tips = [
            "تناول أدويتك يومياً في نفس الموعد المحدد ولا توقف الدواء دون استشارة طبيبك حتى لو شعرت بتحسن.",
            "قم بقياس ضغط الدم في المنزل وسجل القراءات في دفتر المتابعة لمناقشتها مع الطبيب.",
            "تجنب تناول المسكنات ومضادات الالتهاب (مثل الإيبوبروفين والديكلوفيناك) دون إذن الطبيب لأنها قد ترفع الضغط وتؤثر على الكلى.",
        ]
        red_flags = [
            "ألم حاد أو شعور بالضغط والثقل في منتصف الصدر يمتد للذراع الأيسر أو الفك.",
            "صعوبة مفاجئة في الكلام أو اعوجاج في الفم أو ضعف في حركة أحد الذراعين.",
            "صداع شديد جداً ومفاجئ مصحوب بزغللة في العين أو دوار حاد.",
            "ضيق حاد في التنفس أو تورم مفاجئ في الوجه أو الساقين.",
        ]
        emergency_action = "في حال ظهور أي من علامات الخطر المذكورة أعلاه، توجه فوراً إلى أقرب قسم طوارئ أو اتصل بالإسعاف (123)."
    else:
        title = "Patient Health & Blood Pressure Management Guide"
        summary = (
            f"You have been evaluated for {condition_name}. Your target blood pressure goal is {target_bp}. "
            "Controlling your blood pressure protects your heart, brain, and kidneys from long-term complications."
        )
        lifestyle_rules = [
            "Reduce dietary sodium intake to less than 2,000 mg (about 1 teaspoon of table salt) per day.",
            "Adopt the DASH eating plan rich in vegetables, fruits, whole grains, and low-fat dairy.",
            "Engage in at least 150 minutes of moderate aerobic physical activity (e.g. brisk walking) per week.",
            "Complete smoking cessation and avoidance of second-hand smoke.",
            "Maintain a healthy body weight and practice stress-management techniques.",
        ]
        medication_tips = [
            "Take your blood pressure medication consistently every day at the same scheduled time.",
            "Never discontinue or adjust your medication dosage without consulting your physician.",
            "Avoid frequent use of NSAID pain relievers (e.g., ibuprofen, naproxen) as they can elevate blood pressure.",
        ]
        red_flags = [
            "Severe crushing chest pain, tightness, or pressure radiating to the arm, neck, or jaw.",
            "Sudden facial drooping, arm weakness, or difficulty speaking (Stroke warning signs).",
            "Sudden severe headache associated with blurred vision or dizziness.",
            "Acute shortness of breath or sudden swelling of the legs.",
        ]
        emergency_action = "If any of these warning signs occur, call emergency medical services (911 / 123) or go to the nearest emergency department immediately."

    return {
        "title": title,
        "language": language,
        "condition_summary": summary,
        "lifestyle_rules": lifestyle_rules,
        "medication_safety_tips": medication_tips,
        "emergency_red_flags": red_flags,
        "emergency_action": emergency_action,
    }


# --- MULTI-GUIDELINE COMPARISON MATRIX (WHO vs NICE vs ACC/AHA) ---
GUIDELINES_COMPARISON_DATA = [
    {
        "topic": "Diagnostic BP Threshold for Hypertension",
        "who_2021": "Systolic >= 140 mmHg and/or Diastolic >= 90 mmHg on repeated office visits.",
        "nice_ng136": "Stage 1: Clinic >= 140/90 mmHg AND ABPM/HBPM daytime average >= 135/85 mmHg.",
        "acc_aha_2017": "Stage 1: Systolic 130-139 mmHg or Diastolic 80-89 mmHg (Lower threshold).",
        "consensus": "All guidelines agree that treatment is mandatory at >= 140/90 mmHg; ACC/AHA defines pre-hypertension earlier at 130/80.",
    },
    {
        "topic": "First-Line Pharmacological Classes",
        "who_2021": "Three main classes: (1) Thiazide/thiazide-like diuretics, (2) ACEi or ARBs, (3) Calcium Channel Blockers (CCBs).",
        "nice_ng136": "Age < 55 / non-black: ACEi or ARB (Step 1). Age >= 55 or black African/African-Caribbean: CCB (Step 1).",
        "acc_aha_2017": "Primary agents: Thiazide diuretics, CCBs, and ACEi or ARBs (similar to WHO).",
        "consensus": "Universal consensus on the 3 core first-line classes (ACEi/ARB, CCB, Thiazide-like diuretic). Beta-blockers are NOT first-line unless specific compelling indications exist (e.g. post-MI, heart failure).",
    },
    {
        "topic": "Single-Pill Combination (SPC) Strategy",
        "who_2021": "Strong recommendation: Start with two-drug combination therapy (preferably single-pill combination) when baseline BP is >= 20/10 mmHg above target.",
        "nice_ng136": "Stepwise monotherapy first unless baseline BP >= 160/100 mmHg (Step 2 adds second agent).",
        "acc_aha_2017": "Initiate 2 first-line agents of different classes if BP is > 20/10 mmHg above target.",
        "consensus": "Strong shift globally toward early dual combination therapy in a single pill to improve medication adherence.",
    },
    {
        "topic": "Target BP Goal in General Adults",
        "who_2021": "< 140/90 mmHg for all adults; < 130/80 mmHg if known CVD or high cardiovascular risk.",
        "nice_ng136": "Age < 80: Clinic < 140/90 mmHg (ABPM < 135/85). Age >= 80: Clinic < 150/90 mmHg.",
        "acc_aha_2017": "< 130/80 mmHg for all hypertensive adults (more intensive target).",
        "consensus": "Standard target < 140/90 mmHg universally supported; tighter < 130/80 mmHg endorsed when tolerated and in high-risk patients.",
    },
]


def get_guidelines_comparison() -> List[Dict[str, Any]]:
    """Returns the multi-guideline consensus and divergence matrix."""
    return GUIDELINES_COMPARISON_DATA
