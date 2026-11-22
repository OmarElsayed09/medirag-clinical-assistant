"""Prevent unverified medical claims or fictional benchmark results in the UI."""
from pathlib import Path


def test_frontend_does_not_hardcode_unverified_benchmark_scores():
    html = (Path(__file__).resolve().parents[1] / "static/index.html").read_text(encoding="utf-8")
    assert "Zero-Hallucination" not in html
    assert "449 Vectors" not in html
    assert 'id="kpiFaithfulness">95.0%' not in html
    assert 'id="kpiSafetyRefusal">100%' not in html
    assert "No independently verified benchmark run is available." in html
