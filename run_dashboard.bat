@echo off
REM Double-click to start the dashboard. Uses the project's venv if it exists.
cd /d "%~dp0"
set "PY=python"
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    echo No venv found - using the system Python. If it fails, run: pip install -r requirements.txt
)
REM Local use only: allow uploading .pkl files on the Save / Load page.
set "ALLOW_PKL_UPLOAD=1"
echo Starting the dashboard at http://localhost:8501 - close this window to stop it.
REM Open the browser after a few seconds (headless mode skips Streamlit's first-run email question).
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 6; Start-Process 'http://localhost:8501'"
"%PY%" -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
pause
