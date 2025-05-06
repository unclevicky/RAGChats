
@echo off
setlocal
chcp 65001 >nul

:: 设置环境变量
set APP_ENV=development
set PYTHONPATH=..\

:: 启动后端服务
echo [启动服务] 正在启动后端服务...
start "Backend" cmd /k "cd backend && set PYTHONPATH=..\ && uvicorn main:app --reload --port 8000"
timeout /t 3 /nobreak >nul

:: 启动前端服务
echo [启动服务] 正在启动前端服务...
start "Frontend" cmd /k "cd frontend && npm run dev"

:: 完成提示
echo [状态] 前后端服务已启动
echo [访问] 后端: http://localhost:8000
echo [访问] 前端: http://localhost:5173
endlocal
