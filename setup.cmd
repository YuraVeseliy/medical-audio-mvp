@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt -c requirements-lock.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" scripts\setup_asr_models.py
if errorlevel 1 goto fail
echo Подготовка завершена. Запустите launch.cmd
pause
exit /b 0
:fail
echo Подготовка не завершена. Сохраните сообщение об ошибке. Нужны Python 3.12 x64 и интернет для первоначальной установки.
pause
exit /b 1
