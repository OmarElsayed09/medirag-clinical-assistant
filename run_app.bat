@echo off
cd /d "%~dp0"
title MediRAG Clinical Assistant

echo ===================================================
echo   Starting MediRAG Clinical Web Application...
echo   Modern Web SPA: http://localhost:8000
echo   Streamlit UI  : http://localhost:8501
echo ===================================================

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    start python -m uvicorn api:app --port 8000 --host 0.0.0.0
    python -m streamlit run app.py --server.port 8501
) else (
    echo [ERROR] Virtual environment .venv not found.
    echo Please run: python -m venv .venv
    echo And install requirements: pip install -r requirements.txt
)

pause
