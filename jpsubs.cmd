@echo off
rem Launcher for the Japanese immersion tool (Windows).
rem Uses the project's virtualenv if present, otherwise the system Python.
setlocal
set "HERE=%~dp0"

rem Print Japanese correctly even when output is piped or redirected to a file, and make
rem the console itself UTF-8. Without this, Windows falls back to the legacy ANSI codepage
rem and the first transcribed line dies with UnicodeEncodeError.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul 2>&1

if exist "%HERE%.venv\Scripts\python.exe" (
    set "PY=%HERE%.venv\Scripts\python.exe"
    goto :run
)
where /q python.exe && set "PY=python.exe" && goto :run
where /q py.exe && set "PY=py.exe" && goto :run

echo Error: no Python found on PATH and no .venv in "%HERE%".
echo Install Python 3.9+ with:  winget install Python.Python.3.12
echo Then create the environment: see WINDOWS.md
exit /b 1

:run
"%PY%" "%HERE%subgen.py" %*
exit /b %ERRORLEVEL%
