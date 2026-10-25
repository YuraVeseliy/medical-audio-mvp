@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m audio_mvp
) else (
    py -3 -m audio_mvp
)
if errorlevel 1 (
    echo.
    echo Application could not start. Check Python and install project dependencies.
    pause
)
endlocal
