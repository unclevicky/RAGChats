@echo off
setlocal
chcp 65001 >nul

echo [环境检查] 正在检查并激活Conda环境...
call conda activate ragchat_clean
if %errorlevel% neq 0 (
    echo [错误] 无法激活ragchat_clean环境，请先执行以下命令创建环境:
    echo conda create -n ragchat_clean python=3.12 -y
    echo conda activate ragchat_clean
    echo pip install -r backend/requirements.txt
    pause
    exit /b 1
)

echo [环境设置] 正在设置PYTHONPATH环境变量...
cd %~dp0
set PYTHONPATH=%CD%

:: 安装python-multipart包(如果缺少)
pip show python-multipart >nul 2>&1
if %errorlevel% neq 0 (
    echo [依赖安装] 正在安装python-multipart包...
    pip install python-multipart
)

:: 创建必要的目录
if not exist backend\logs mkdir backend\logs
if not exist backend\data mkdir backend\data
if not exist backend\meta mkdir backend\meta
if not exist backend\vectorstore mkdir backend\vectorstore

:: 启动前端服务
echo [启动服务] 正在启动前端服务...
start "RAGChats Frontend" cmd /k "cd frontend && npm run dev"
timeout /t 2 /nobreak >nul

:: 启动后端服务
echo [启动服务] 正在启动后端服务...
start "RAGChats Backend" cmd /k "cd backend && set PYTHONPATH=%CD%\.. && python -m uvicorn main:app --reload"

echo.
echo [服务启动完成]
echo 你可以通过以下地址访问服务:
echo 前端服务: http://localhost:5173
echo 后端API: http://localhost:8000
echo.
echo 按任意键退出此窗口，服务将继续在后台运行...
pause >nul
endlocal 