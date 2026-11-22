# Validation and limitations

MediRAG is a portfolio prototype and has **not** been independently validated for patient care.

## Before claiming clinical reliability

1. Version the exact guideline PDFs, their dates, and licenses.
2. Create a clinician-reviewed question set with expected citations, abstention cases, and adversarial examples.
3. Report retrieval metrics, answer faithfulness, citation accuracy, and risk-stratified clinical safety with confidence intervals.
4. Run security checks for prompt injection, authorization, PHI handling, and audit logs.
5. Document human review requirements and intended-use exclusions.

A low LLM temperature and a closed-domain prompt do **not** prove zero hallucination. Current code and available tests do not establish a 100% safety-refusal rate.

Do not use this application to diagnose, prescribe, or manage patients.
