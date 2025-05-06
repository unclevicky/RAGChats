@echo off
:: 设置UTF-8编码支持中文显示
chcp 65001 > nul
setlocal enabledelayedexpansion

:: 设置颜色变量
for /f "delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do set "DEL=%%a"
set "RED=%DEL%[91m"
set "GREEN=%DEL%[92m"
set "YELLOW=%DEL%[93m"
set "BLUE=%DEL%[94m"
set "RESET=%DEL%[0m"

title 前端项目打包工具

echo %BLUE%=== 前端打包流程 ===%RESET%
echo.

:: 第一步：构建Vue生产版本
echo %YELLOW%第一步：构建Vue生产版本%RESET%
call build_vue.bat
if %errorlevel% neq 0 (
    echo %RED%错误：Vue构建失败%RESET%
    pause
    exit /b 1
)

:: 第二步：使用electron-builder打包
echo %YELLOW%第二步：使用electron-builder打包%RESET%
cd frontend
npm run electron:build
if %errorlevel% neq 0 (
    echo %RED%错误：应用打包失败%RESET%
    pause
    exit /b 1
)

:: 完成提示
echo.
echo %GREEN%=== 打包成功 ===%RESET%
echo 生成的可执行文件位于 frontend\dist 目录下
echo.
pause