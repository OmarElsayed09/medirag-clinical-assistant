# Reproducible quality evidence

## What GitHub Actions verifies

`clinical-unit-tests.yml` runs the existing clinical calculator and security-rule unit tests, with Python 3.11 and a minimal pytest installation. Each run saves a JUnit XML artifact.

The tests inspect code behavior against written assertions. Passing **does not** prove clinical accuracy, medical device compliance, safety for patients, or robustness against unseen attacks.

## Reproduce locally

```bash
python -m venv .venv
# Activate your environment, then:
python -m pip install "pytest>=8,<10"
python -m pytest -q tests/test_clinical_tools.py tests/test_security.py --junitxml=artifacts/clinical-tests.xml
```

For full application integration or RAG benchmark work, install `requirements.txt` and configure supported LLM/embedding providers. Record reference dataset version, guideline source date, human labels, model version, retriever settings, and run SHA before publishing any score.

## Demo capture

Run the app against the documented local setup. Capture **real UI screenshots** only after the interface launches, and redact any identifiers or patient details. Save captures under `docs/assets/` and label them as prototype demonstrations. No synthetic application screenshot or outcome is included.
