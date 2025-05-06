@echo off
REM 打包Python后端为可执行文件
echo Building Python executable...

REM 确保使用虚拟环境（推荐）
REM python -m venv venv
REM call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt  # 假设你有 requirements.txt
pip install pyinstaller

REM 执行打包命令
pyinstaller --onedir ^
    --add-data "backend/model_cache;model_cache" ^
    --add-data "backend/middleware.py;." ^
    --add-data "E:/soft/tools/anaconda3/envs/llm_learn/Lib/site-packages/zh_core_web_sm;zh_core_web_sm" ^
    --hidden-import "pydantic" ^
    --hidden-import "pydantic.main" ^
    --hidden-import "pydantic.networks" ^
    --hidden-import "pydantic.types" ^
    --hidden-import "pydantic.json" ^
    --hidden-import "sentence_transformers" ^
    --hidden-import "llama_index" ^
    --hidden-import "tiktoken" ^
    --hidden-import "tiktoken_ext" ^
    --hidden-import "tiktoken_ext.openai_public" ^
    --hidden-import "spacy" ^
    --hidden-import "spacy.pipeline" ^
    --hidden-import "spacy_pkuseq" ^
    --hidden-import "uvicorn" ^
    --hidden-import "asyncio" ^
    --hidden-import "httpx" ^
    --hidden-import "starlette" ^
    --hidden-import "starlette.middleware.base" ^
    --hidden-import "fastapi" ^
    backend\main.py

echo Python executable built in dist folder
pause