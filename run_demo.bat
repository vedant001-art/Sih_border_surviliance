@echo off
echo.
echo ============================================
echo   AI Border Surveillance Platform (SIH26187)
echo ============================================
echo.

echo [1/2] Cleaning up old processes on port 8000...
taskkill /F /FI "WINDOWTITLE eq BorderSec*" >NUL 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (taskkill /F /PID %%a >NUL 2>&1)
timeout /t 2 /nobreak > NUL

echo [2/2] Starting Border Surveillance Platform...
start "BorderSec Backend" cmd /k "cd /d %~dp0 && set FLAGS_use_mkldnn=0 && set FLAGS_enable_pir_api=0 && set OMP_NUM_THREADS=2 && %~dp0venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo      Waiting for AI models to initialize (~15 seconds)...
timeout /t 15 /nobreak > NUL

echo.
echo ============================================
echo   System started successfully!
echo   Opening http://127.0.0.1:8000 in your browser
echo ============================================
echo.
echo   NOTE: Keep the "BorderSec Backend" window open.
echo.
start http://127.0.0.1:8000
pause
