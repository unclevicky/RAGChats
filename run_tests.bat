
@echo off
setlocal
set APP_ENV=test
set PYTHONPATH=..\
cd backend
pytest -v tests/
endlocal
