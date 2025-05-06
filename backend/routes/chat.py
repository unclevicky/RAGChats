
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import json
import sys
import os
import asyncio
from pathlib import Path
import logging
from backend.utils import load_vector_index, create_rag_engine
# from backend.config import MODEL_CONFIG
def get_config_path():
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return Path(base_path) / "config.json"
CONFIG_PATH = get_config_path()
def load_model_config():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载模型配置失败: {str(e)}")
        return {}
MODEL_CONFIG = load_model_config()

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='backend/logs/backend.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

router = APIRouter()

async def generate_chat_response(assistant: Dict[str, Any], question: str, history: List[Dict[str, Any]] = None):
    """生成聊天响应流"""
    try:
        logger.info(f"开始处理聊天请求，助手完整配置: {json.dumps(assistant, indent=2, ensure_ascii=False)}")
        logger.info(f"问题内容: {question}")
        logger.info(f"历史记录内容: {json.dumps(history, indent=2, ensure_ascii=False)}")

        # 验证助手配置
        required_keys = ['knowledge_base', 'model', 'name', 'description', 'system_prompt']
        for key in required_keys:
            if key not in assistant:
                raise ValueError(f"助手配置缺少必要字段: {key}")
        
        # 加载向量索引
        kb_path = assistant["knowledge_base"]
        logger.info(f"正在加载知识库向量索引，知识库ID: {kb_path}")
        logger.info(f"使用的嵌入模型: {assistant.get('embedding', 'text-embedding-3-small')}")
        try:
            vector_index = load_vector_index(
                kb_path,
                embedding_model_id=assistant.get("embedding", "text-embedding-3-small")
            )
        except Exception as e:
            logger.error(f"加载向量索引失败，知识库路径: {kb_path}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"加载知识库失败，请确保知识库'{kb_path}'已正确初始化"
            )
        
        # 创建RAG引擎
        model_id = assistant["model"]
        logger.info(f"正在创建RAG引擎，使用模型ID: {model_id}")
        logger.info(f"模型配置: {json.dumps(MODEL_CONFIG.get(model_id, {}), indent=2, ensure_ascii=False)}")
        rag_engine = create_rag_engine(vector_index, model_id)
        logger.info("RAG引擎创建成功")
        
        # 构建上下文提示
        context_prompt = f"""
        你是一个智能助手，以下是你的配置信息：
        - 名称: {assistant["name"]}
        - 描述: {assistant["description"]}
        - 系统提示词: {assistant["system_prompt"]}
        
        以下是历史对话记录：
        {format_history(history)}
        """
        
        # 构建完整问题
        full_question = f"{context_prompt}\n用户提问: {question}"
        logger.debug(f"完整问题上下文: {full_question}")
        
        # 获取响应流
        logger.debug("正在获取RAG响应...")
        response = rag_engine.query(full_question)
        
        # 流式返回响应 (使用SSE格式)
        async def generate():
            try:
                for chunk in response.response_gen:
                    # 按照SSE格式发送数据
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
            except Exception as e:
                logger.error(f"生成流式响应时出错: {str(e)}", exc_info=True)
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(0)  # 添加这行关键代码
                
        logger.debug("成功生成聊天响应流")
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 添加强制禁用缓冲头
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        logger.error(f"生成聊天响应时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def format_history(history: List[Dict[str, Any]]) -> str:
    """格式化历史对话记录"""
    if not history:
        return "无历史对话"
    
    formatted = []
    for msg in history:
        role = "助手" if msg["type"] == "bot" else "用户"
        formatted.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted)

@router.post("/chat")
async def chat_endpoint(data: Dict[str, Any]):
    """处理聊天请求"""
    try:
        logger.info("收到新的聊天请求，原始数据: %s", json.dumps(data, indent=2, ensure_ascii=False))
        assistant = data.get("assistant")
        question = data.get("question")
        history = data.get("history", [])
        
        if assistant is None:
            logger.error("请求中缺少assistant参数")
        if question is None:
            logger.error("请求中缺少question参数")
        
        logger.info("助手信息: %s", assistant)
        logger.info("问题: %s", question)
        logger.info("历史记录: %s", history)
        if not assistant or not question:
            logger.warning("请求缺少必要参数")
            raise HTTPException(status_code=400, detail="缺少必要参数")
            
        logger.debug("参数验证通过，开始生成响应")
        return await generate_chat_response(assistant, question, history)
        
    except Exception as e:
        logger.error(f"处理聊天请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
