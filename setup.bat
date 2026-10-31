@echo off
REM Launcher for setup.ps1 - bypasses PowerShell execution policy for this run only.
REM Pass-through option: -Recreate
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
pause
