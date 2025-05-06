@echo off
REM Ensure running in CMD
ver >nul 2>&1 || (
    echo ERROR: This script requires Command Prompt (CMD)
    exit /b 1
)

REM Clean previous build
if exist "frontend\dist" (
    rmdir /s /q "frontend\dist"
)

REM Install dependencies
echo [1/3] Installing dependencies...
cd frontend
npm install
if errorlevel 1 (
    echo ERROR: Dependency installation failed
    cd ..
    exit /b 1
)

REM Build production
echo [2/3] Building production version...
npm run build
if errorlevel 1 (
    echo ERROR: Build failed
    cd ..
    exit /b 1
)
cd ..

REM Verify output
echo [3/3] Verifying build output...
if not exist "frontend\dist\index.html" (
    echo ERROR: Build output missing
    exit /b 1
)

echo SUCCESS: Vue build completed
echo Output: frontend\dist
pause