"""
Integration and API endpoint tests for MediRAG FastAPI backend (api.py).
Tests:
- /health and /api/status endpoints
- /tools/bp-stage endpoint
- /tools/renal-calc endpoint
- /tools/cvd-risk endpoint
- /expert/guidelines-comparison endpoint
- /expert/drug-interactions endpoint
- /documents list endpoint
"""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


class TestHealthAndStatusAPI:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_root_status_endpoint(self):
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "MediRAG Clinical Assistant API"
        assert "vector_store_chunks" in data
        assert "disclaimer" in data


class TestClinicalToolsAPI:
    def test_bp_stage_endpoint(self):
        payload = {
            "systolic": 150,
            "diastolic": 95,
            "has_diabetes": False,
        }
        response = client.post("/tools/bp-stage", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "Stage 1" in data["stage"]

    def test_bp_stage_invalid(self):
        payload = {
            "systolic": 40,
            "diastolic": 20,
        }
        response = client.post("/tools/bp-stage", json=payload)
        assert response.status_code == 422 or response.status_code == 400

    def test_renal_calc_endpoint(self):
        payload = {
            "serum_creatinine_mg_dl": 1.1,
            "age_years": 55,
            "is_female": False,
            "weight_kg": 75.0,
        }
        response = client.post("/tools/renal-calc", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "egfr_ckd_epi" in data
        assert "kdigo_stage" in data

    def test_cvd_risk_endpoint(self):
        payload = {
            "age": 60,
            "systolic_bp": 150,
            "is_smoker": True,
            "has_diabetes": True,
        }
        response = client.post("/tools/cvd-risk", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "risk_tier" in data


class TestClinicalExpertAPI:
    def test_guidelines_comparison(self):
        response = client.get("/expert/guidelines-comparison")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "topic" in data[0]

    def test_drug_interactions_endpoint(self):
        payload = {
            "medications": ["ace_inhibitor", "spironolactone", "nsaid"]
        }
        response = client.post("/expert/drug-interactions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_interactions_found"] >= 1
        assert "interactions" in data

    def test_export_soap_pdf_endpoint(self):
        payload = {
            "age": 60,
            "is_female": False,
            "systolic_bp": 155,
            "diastolic_bp": 95,
            "has_diabetes": True,
            "serum_creatinine_mg_dl": 1.3,
            "current_medications": ["ace_inhibitor", "spironolactone"],
        }
        response = client.post("/expert/export-soap-pdf", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content[:5] == b"%PDF-"

    def test_export_patient_guide_pdf_endpoint(self):
        payload = {
            "condition_name": "Stage 2 Hypertension",
            "target_bp": "< 130/80 mmHg",
            "language": "English",
        }
        response = client.post("/expert/export-patient-guide-pdf", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content[:5] == b"%PDF-"


class TestDocumentRepositoryAPI:
    def test_list_documents(self):
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "total_files" in data
        assert "documents" in data
