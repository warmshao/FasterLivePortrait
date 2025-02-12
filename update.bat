@echo off
git fetch origin
git reset --hard origin/master

".\venv\python.exe" -c "import pip;  try: pip.main(['config', 'unset', 'global.proxy']) except Exception: pass"
".\venv\python.exe" -m pip install -r .\requirements_win.txt
pause