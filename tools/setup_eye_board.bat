@echo off
rem Runs the OpenIris setup menu (name, Wi-Fi, mode) on a board over USB.
rem Usage: double-click (finds the plugged-in XIAO by its USB id 303A), or: setup_eye_board.bat COM5
set PORT=%1
if "%PORT%"=="" for /f %%p in ('powershell -NoProfile -Command "(Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -match 'VID_303A' -and $_.FriendlyName -match '\((COM\d+)\)' } | ForEach-Object { $Matches[1] }) -join ' '"') do set PORT=%%p
if "%PORT%"=="" (
    echo No XIAO board found. Plug one in and try again.
    pause
    exit /b 1
)
echo Using %PORT%
"%USERPROFILE%\.espressif\python_env\idf5.4_py3.13_env\Scripts\python.exe" "%~dp0..\repos\OpenIris-ESPIDF\tools\setup_openiris.py" --port %PORT%
pause
