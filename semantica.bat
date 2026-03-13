@echo off
echo Starting Semantica...
echo.
cd /d "%~dp0semantica"
C:/Users/kko8/AppData/Local/Programs/Python/Python311/python.exe -m uvicorn semantica:app --host 127.0.0.1 --port 5000
