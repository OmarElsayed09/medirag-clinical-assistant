@echo off
cd /d "%~dp0"
title MediRAG Clinical Decision Platform

echo ===================================================
echo   Starting MediRAG Enterprise Clinical Platform...
echo   Web Application: http://localhost:8000
echo   API Swagger Doc: http://localhost:8000/docs
echo ===================================================

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    start http://localhost:8000
    python -m uvicorn api:app --port 8000 --host 0.0.0.0 --reload
) else (
    echo [ERROR] Virtual environment .venv not found.
    echo Please run: python -m venv .venv
    echo And install requirements: pip install -r requirements.txt
    pause
)
