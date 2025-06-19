@echo off
REM 设置环境变量
set PYTHONPATH=%CD%

REM 启动后端服务
start "RAGChats Backend" cmd /k "cd backend && set PYTHONPATH=%CD%\.. && uvicorn main:app --reload --port 8000"

REM 启动前端服务
start "RAGChats Frontend" cmd /k "cd frontend && npm run dev"

echo [状态] 前后端服务已启动
echo [访问] 后端: http://localhost:8000
echo [访问] 前端: http://localhost:5173 