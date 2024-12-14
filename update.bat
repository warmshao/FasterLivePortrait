@echo off
git fetch origin
git reset --hard origin/main
"venv\Scripts\pip.exe" install -r requirements.txt
pause