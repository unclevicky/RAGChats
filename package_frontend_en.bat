@echo off
chcp 936 > nul
setlocal enabledelayedexpansion

echo === Frontend Packaging Process ===
echo [%date% %time%] Process started

echo Step 1: Building Vue production
call "%~dp0build_vue.bat"
if errorlevel 1 (
    echo [ERROR] Vue build failed with error code !errorlevel!
    goto error
)
echo [OK] Vue build completed successfully

echo Step 2: Packaging with electron-builder
cd /d "%~dp0frontend"
if errorlevel 1 (
    echo [ERROR] Failed to change to frontend directory
    goto error
)

echo [INFO] Running electron-builder...
npm run electron:build
if errorlevel 1 (
    echo [ERROR] Electron build failed with error code !errorlevel!
    goto error
)

echo === Packaging Successful ===
echo [OK] Executable is located in: frontend\dist
echo [%date% %time%] Process completed
pause
exit /b 0

:error
echo [ERROR] Packaging process failed at step !errorstep!
echo [%date% %time%] Process terminated with errors
pause
exit /b 1