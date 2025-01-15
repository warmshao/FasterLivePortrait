@echo off
git fetch origin
git reset --hard origin/master
".\venv\python.exe" -m pip config unset global.proxy first
".\venv\Scripts\pip.exe" install -r requirements_win.txt
pause