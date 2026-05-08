@echo off
echo Launching 3 TMRL Workers...

start cmd /k "cd /d "%~dp0" && .venv\Scripts\activate.bat && python -m tmrl --worker"
timeout /t 2 /nobreak > NUL

start cmd /k "cd /d "%~dp0" && .venv\Scripts\activate.bat && python -m tmrl --worker"
timeout /t 2 /nobreak > NUL

start cmd /k "cd /d "%~dp0" && .venv\Scripts\activate.bat && python -m tmrl --worker"

echo All workers launched in separate windows!
