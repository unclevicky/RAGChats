@echo off
echo 开始整合发布包...

REM 创建发布目录
if not exist "rag_chat_dist" mkdir "rag_chat_dist"

REM 复制后端打包结果
if exist "dist\main.exe" (
    if not exist "rag_chat_dist\backend" mkdir "rag_chat_dist\backend"
    copy "dist\main.exe" "rag_chat_dist\backend\"
) else (
    echo 错误：未找到后端打包结果
    pause
    exit /b 1
)

REM 复制前端打包结果
if exist "frontend\dist\win-unpacked" (
    if not exist "rag_chat_dist\frontend" mkdir "rag_chat_dist\frontend"
    xcopy /e /i "frontend\dist_electron\win-unpacked" "rag_chat_dist\frontend\"
) else (
    echo 错误：未找到前端打包结果
    pause
    exit /b 1
)

REM 复制模型缓存
if exist "backend\model_cache" (
    if not exist "rag_chat_dist\model_cache" mkdir "rag_chat_dist\model_cache"
    xcopy /e /i "backend\model_cache" "rag_chat_dist\model_cache\"
)

REM 复制新需求的目录和文件
REM 复制 backend/data
if exist "backend\data" (
    if not exist "rag_chat_dist\backend\data" mkdir "rag_chat_dist\backend\data"
    xcopy /e /i "backend\data" "rag_chat_dist\backend\data\"
)

REM 复制 backend/logs
if exist "backend\logs" (
    if not exist "rag_chat_dist\backend\logs" mkdir "rag_chat_dist\backend\logs"
    xcopy /e /i "backend\logs" "rag_chat_dist\backend\logs\"
)

REM 复制 backend/meta
if exist "backend\meta" (
    if not exist "rag_chat_dist\backend\meta" mkdir "rag_chat_dist\backend\meta"
    xcopy /e /i "backend\meta" "rag_chat_dist\backend\meta\"
)

REM 复制 backend/vectorstore
if exist "backend\vectorstore" (
    if not exist "rag_chat_dist\backend\vectorstore" mkdir "rag_chat_dist\backend\vectorstore"
    xcopy /e /i "backend\vectorstore" "rag_chat_dist\backend\vectorstore\"
)

REM 复制 backend/config.py
if exist "backend\config.py" (
    copy "backend\config.py" "rag_chat_dist\backend\"
)

REM 复制 backend/assistants.py
if exist "backend\assistants.py" (
    copy "backend\assistants.py" "rag_chat_dist\backend\"
)

echo 发布包整合完成，位于 rag_chat_dist 目录下
pause