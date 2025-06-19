from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path
import os
import logging
from typing import Dict
import asyncio
import time

def get_project_root():
    if getattr(sys, 'frozen', False):
        # 打包环境：使用临时解压目录或EXE所在目录
        if hasattr(sys, '_MEIPASS'):
            return sys._MEIPASS # 单文件模式临时目录
        else:
            return os.path.dirname(sys.executable) # 单文件夹模式EXE目录
    else:
        # 开发环境：基于__file__计算路径
        return os.path.dirname(os.path.abspath(__file__))

# 并不是project_root,而是backend的路径
project_root = get_project_root()

sys.path.insert(0, str(Path(project_root)))
from backend.routes import system, chat, knowledge
from backend.utils import process_single_file, process_documents
from fastapi.middleware.cors import CORSMiddleware
from backend.middleware import StreamBufferMiddleware

# 全局变量
# 这种方式不能兼顾打包之后exe运行模式
# DATA_DIR = Path(__file__).parent / "data"
# META_DIR = Path(__file__).parent / "meta"
DATA_DIR = Path(project_root) / "data"
META_DIR = Path(project_root) / "meta"


def setup_logs_dir():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)  # 使用os创建目录
    return log_dir

def setup_logging():
    log_dir = setup_logs_dir()
    log_file = os.path.join(log_dir, "backend.log")
    
    # 创建双重处理器（文件 + 控制台）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # 统一格式
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 重置并配置全局处理器
    logging.basicConfig(
        level=logging.INFO,  # 默认使用INFO级别
        handlers=[file_handler, console_handler],  # 同时输出到文件和控制台
        force=True
    )
    
    # 设置特定模块的日志级别
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('llama_index').setLevel(logging.WARNING)
    
    # 添加日志轮转
    from logging.handlers import RotatingFileHandler
    rotating_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    rotating_handler.setFormatter(formatter)
    
    # 替换之前的文件处理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and not isinstance(handler, RotatingFileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(rotating_handler)
    
    # 记录启动日志
    logging.info("=" * 50)
    logging.info("RAGChats 后端服务启动")
    logging.info("=" * 50)

# 在应用创建前设置日志
setup_logging()

def create_app():
    app = FastAPI(
        title="RAGChats API",
        description="RAG知识库聊天系统API",
        version="1.0.0"
    )

    # 注册请求跟踪中间件
    from backend.middleware import RequestTrackingMiddleware
    app.add_middleware(RequestTrackingMiddleware, slow_request_threshold=3.0)
    
    # 注册并发限制中间件
    from backend.middleware import ConcurrencyLimiterMiddleware
    app.add_middleware(
        ConcurrencyLimiterMiddleware,
        chat_limit=10,         # 最多同时处理10个聊天请求
        kb_creation_limit=3,   # 最多同时处理3个知识库创建请求
        file_process_limit=5   # 最多同时处理5个文件处理请求
    )
    
    # 注册流媒体中间件
    from backend.middleware import StreamBufferMiddleware
    app.add_middleware(StreamBufferMiddleware)
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # 允许的前端地址
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有方法
        allow_headers=["*"],  # 允许所有头
    )
    
    # 注册错误处理器
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        logging.error(f"HTTP异常: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc):
        logging.exception(f"未处理的异常: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={"error": "服务器内部错误，请查看日志了解详情"},
        )
    
    # 注册路由
    app.include_router(knowledge.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(system.router, prefix="/api/system")
    
    # 注册启动和关闭事件
    @app.on_event("startup")
    async def startup_event():
        logging.info("应用启动中...")
        # 创建必要的目录
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(META_DIR, exist_ok=True)
        logging.info("应用启动完成")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logging.info("应用关闭中...")
        # 清理资源
        logging.info("应用已关闭")
    
    return app

app = create_app()

# 文件处理相关接口
@app.post("/knowledge-bases/{kb_id}/files/{filename}/process")
async def process_file(
    kb_id: str, 
    filename: str,
    payload: Dict = Body(default={}),
    embedding_model_id: str = None,
    incremental: bool = True
):
    """处理知识库中的单个文件"""
    # 参数处理和验证逻辑...
    # 保留原有实现...
    """处理知识库中的单个文件"""
    # 参数处理和验证逻辑...
    # 保留原有实现...

@app.post("/knowledge-bases/{kb_id}/process-batch")
async def process_batch(
    kb_id: str,
    embedding_model_id: str = "bge-large-zh-v1.5",
    incremental: bool = True,
    max_workers: int = 4
):
    """批量处理知识库中的文件"""
    # 保留原有实现...
    """批量处理知识库中的文件"""
    # 保留原有实现...

# RAG核心API
@app.post("/knowledge-bases/{kb_id}/index")
async def index_knowledge_base(kb_id: str):
    """为知识库创建向量索引"""
    return {"kb_id": kb_id, "status": "indexing", "progress": 0}

@app.post("/query")
async def query_knowledge_base(query: str, kb_id: str = None):
    """查询知识库"""
    return {
        "query": query,
        "kb_id": kb_id,
        "results": [],
        "answer": "This is a placeholder response. RAG functionality will be implemented later."
    }

if __name__ == "__main__":
    import uvicorn
    from backend import utils
    from backend.routes import system, chat, knowledge
    uvicorn.run(app, host="0.0.0.0", port=8000)
