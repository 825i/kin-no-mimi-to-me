@echo off
rem Launcher for the Japanese immersion tool (Windows).
rem Uses the project's virtualenv if present, otherwise the system Python.
setlocal
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\python.exe" (
    set "PY=%HERE%.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" "%HERE%subgen.py" %*
