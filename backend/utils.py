import os
import sys
import json
import logging
from pathlib import Path
import faiss
import shutil
import tempfile
from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext, ServiceContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.node_parser import SentenceSplitter
from backend.knowledge_meta import KnowledgeMetaManager, KnowledgeBaseMeta, DocumentMeta
from datetime import datetime
import concurrent.futures
import threading
import time
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple
from langchain.text_splitter import SpacyTextSplitter
from llama_index.node_parser import LangchainNodeParser

meta_manager = KnowledgeMetaManager()

# 全局缓存，用于存储已加载的向量索引和RAG引擎
# 使用LRU缓存以避免内存溢出
# 格式: {kb_id+embedding_model_id: (index, last_access_time)}
_VECTOR_INDEX_CACHE: Dict[str, Tuple[Any, float]] = {}
_VECTOR_INDEX_CACHE_LOCK = threading.RLock()
_VECTOR_INDEX_CACHE_MAX_SIZE = 5  # 最多缓存5个向量索引

# RAG引擎缓存
# 格式: {kb_id+embedding_model_id+chat_model_id: (engine, last_access_time)}
_RAG_ENGINE_CACHE: Dict[str, Tuple[Any, float]] = {}
_RAG_ENGINE_CACHE_LOCK = threading.RLock()
_RAG_ENGINE_CACHE_MAX_SIZE = 10  # 最多缓存10个RAG引擎

# 获取配置路径
def get_config_path():
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return Path(base_path) / "config.json"

CONFIG_PATH = get_config_path()

# 加载模型配置
def load_model_config():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载模型配置失败: {str(e)}")
        return {}

# 默认模型配置
DEFAULT_MODEL_CONFIG = {
    "bge-large-zh-v1.5": {
        "model": "BAAI/bge-large-zh-v1.5"
    },
    "huggingface_bge-large-zh-v1.5": {
        "model": "BAAI/bge-large-zh-v1.5"
    }
}

# 尝试从配置文件加载，如果失败则使用默认配置
MODEL_CONFIG = load_model_config() or DEFAULT_MODEL_CONFIG

# 获取项目根目录
def get_project_root():
    if getattr(sys, 'frozen', False):
        # 打包环境：使用临时解压目录或EXE所在目录
        if hasattr(sys, '_MEIPASS'):
            return sys._MEIPASS  # 单文件模式临时目录
        else:
            return os.path.dirname(sys.executable)  # 单文件夹模式EXE目录
    else:
        # 开发环境：基于__file__计算路径
        return os.path.dirname(os.path.abspath(__file__))

# 并不是project_root,而是backend的路径
project_root = get_project_root()

def process_documents(file_path, vector_path, embedding_model_id, incremental=True, kb_id=None, max_workers=1):
    """处理文档并创建/更新向量索引，支持并发处理"""
    process_lock = threading.Lock()
    os.makedirs(vector_path, exist_ok=True)
    os.makedirs(file_path, exist_ok=True)
    logging.info(f"1.确认知识库 {kb_id}，原始文件路径: {file_path}，向量存储路径: {vector_path}, 嵌入模型: {embedding_model_id}")

    # 使用线程本地存储，避免并发冲突
    thread_local = threading.local()
    
    kb_meta = meta_manager.get_knowledge_base(kb_id) if kb_id else None
    if not kb_meta:
        kb_meta = KnowledgeBaseMeta(
            kb_id=kb_id or os.path.basename(vector_path),
            source_path=file_path,
            vector_path=vector_path,
            embedding_model=embedding_model_id,
            incremental=incremental
        )
    
    # 设置环境变量以避免OpenAI相关错误
    os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
    
    try:
        loader = SimpleDirectoryReader(file_path)
        all_documents = loader.load_data()
        logging.info(f"共读取到 {len(all_documents)} 个文档")
        
        # 更新文档元数据状态
        for doc in all_documents:
            doc_id = os.path.basename(doc.metadata.get('file_path', ''))
            if doc_id:
                doc_meta = kb_meta.documents.get(doc_id, DocumentMeta(doc_id=doc_id, name=doc_id))
                doc_meta.status = "处理中"
                kb_meta.documents[doc_id] = doc_meta
        
        # 初始化嵌入模型 - 使用全局锁确保单例
        embedding_model_lock = threading.Lock()
        with embedding_model_lock:
            embedding_model = _initialize_embedding_model(embedding_model_id)
        
        # 替换为LangchainNodeParser+SpacyTextSplitter
        text_splitter = LangchainNodeParser(
            SpacyTextSplitter(
                pipeline="zh_core_web_sm",
                chunk_size=1000,
                chunk_overlap=200
            )
        )
        
        logging.info(f"2.初始化完成,处理文件 {file_path}，向量存储路径 {vector_path}，嵌入模型ID {embedding_model_id}，增量更新 {incremental}，知识库ID {kb_id}")
        
        # 验证向量存储目录可写
        try:
            os.makedirs(vector_path, exist_ok=True)
            test_file = os.path.join(vector_path, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            error_msg = f"无法写入向量存储目录 {vector_path}: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # 初始化向量存储
        dimension = 1024
        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 服务上下文，显式设置llm=None避免使用OpenAI
        service_context = ServiceContext.from_defaults(
            embed_model=embedding_model,
            text_splitter=text_splitter,
            llm=None
        )
        
        # 只单线程处理，不分批，所有文档一次性向量化
        try:
            index = VectorStoreIndex.from_documents(
                all_documents,
                storage_context=storage_context,
                service_context=service_context
            )
            # 更新文档状态
            for doc in all_documents:
                doc_id = os.path.basename(doc.metadata.get('file_path', ''))
                if doc_id:
                    doc_meta = kb_meta.documents.get(doc_id, DocumentMeta(doc_id=doc_id, name=doc_id))
                    doc_meta.status = "已完成"
                    doc_meta.last_updated = datetime.now().isoformat()
                    kb_meta.documents[doc_id] = doc_meta
            results = [True]
        except Exception as e:
            logging.error(f"文档向量化失败: {str(e)}")
            for doc in all_documents:
                doc_id = os.path.basename(doc.metadata.get('file_path', ''))
                if doc_id:
                    doc_meta = kb_meta.documents.get(doc_id, DocumentMeta(doc_id=doc_id, name=doc_id))
                    doc_meta.status = "失败"
                    doc_meta.error = str(e)
                    kb_meta.documents[doc_id] = doc_meta
            results = [False]
            raise
        # 保存向量索引
        index.storage_context.persist(persist_dir=vector_path)
        
        # 更新知识库元数据
        kb_meta.last_processed = datetime.now().isoformat()
        kb_meta.processed_files = len(all_documents)
        kb_meta.update_time = datetime.now()
        meta_manager.save_knowledge_base(kb_meta)
        
        logging.info(f"3.向量化处理完成,处理文件 {file_path}，向量存储路径 {vector_path}，嵌入模型ID {embedding_model_id}，增量更新 {incremental}，知识库ID {kb_id}")
        return index, results
        
    except Exception as e:
        # 更新所有文档状态为失败
        for doc in all_documents:
            doc_id = os.path.basename(doc.metadata.get('file_path', ''))
            if doc_id:
                doc_meta = kb_meta.documents.get(doc_id, DocumentMeta(doc_id=doc_id, name=doc_id))
                doc_meta.status = "失败"
                doc_meta.error = str(e)
                kb_meta.documents[doc_id] = doc_meta
        kb_meta.last_processed = datetime.now().isoformat()
        kb_meta.processed_files = 0
        kb_meta.update_time = datetime.now()
        meta_manager.save_knowledge_base(kb_meta)
        error_msg = f"处理文件 {file_path} 失败: {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg)

def _initialize_embedding_model(embedding_model_id):
    """初始化嵌入模型，抽取为独立函数便于复用"""
    embedding_config = MODEL_CONFIG.get(embedding_model_id)
    if not embedding_config or 'model' not in embedding_config:
        raise ValueError(f"未找到模型配置或缺少model参数: {embedding_model_id}")
    
    model_name = str(embedding_config['model'])
    
    # 1. 检查backend/model_cache目录
    local_model_path = Path(__file__).parent / "model_cache" / model_name
    logging.info(f"尝试使用本地模型路径1: {local_model_path}")
    
    # 2. 检查项目根目录下的model_cache目录
    if not local_model_path.exists():
        local_model_path = Path(__file__).parent.parent / "model_cache" / model_name
        logging.info(f"尝试使用本地模型路径2: {local_model_path}")
        
        # 3. 检查用户home目录下的.cache/huggingface目录
        if not local_model_path.exists():
            # 尝试系统默认缓存路径
            default_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
            logging.info(f"尝试使用HuggingFace缓存路径: {default_cache}")
            if default_cache.exists():
                # 找到最新的snapshot
                snapshots = list(default_cache.glob("snapshots/*"))
                if snapshots:
                    local_model_path = snapshots[-1]
                    logging.info(f"找到HuggingFace缓存模型: {local_model_path}")
    
    # 4. 检查是否需要创建目录
    model_dir = Path(__file__).parent / "model_cache" / "BAAI"
    if not model_dir.exists():
        try:
            os.makedirs(model_dir, exist_ok=True)
            logging.info(f"创建模型缓存目录: {model_dir}")
        except Exception as e:
            logging.warning(f"创建模型缓存目录失败: {str(e)}")
    
    # 创建嵌入模型实例 - 添加重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if local_model_path.exists():
                logging.info(f"使用本地模型路径: {local_model_path}")
                embedding_model = HuggingFaceEmbedding(
                    model_name=str(local_model_path),
                    device="cpu",
                    trust_remote_code=True
                )
            else:
                logging.warning(f"本地模型未找到，尝试在线下载: {model_name}")
                embedding_model = HuggingFaceEmbedding(
                    model_name=model_name,
                    device="cpu",
                    trust_remote_code=True
                )
            
            # 测试模型是否正常工作
            test_embedding = embedding_model.get_text_embedding("测试")
            if len(test_embedding) > 0:
                logging.info(f"嵌入模型初始化成功，向量维度: {len(test_embedding)}")
                return embedding_model
            else:
                raise ValueError("嵌入模型返回空向量")
                
        except Exception as e:
            logging.error(f"嵌入模型初始化失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise ValueError(f"嵌入模型初始化失败: {str(e)}")
            time.sleep(1)  # 等待1秒后重试
    
    raise ValueError("嵌入模型初始化失败，已达到最大重试次数")

def load_vector_index(knowledge_base_id, embedding_model_id: str = "bge-large-zh-v1.5"):
    """加载向量索引，优先从缓存中读取"""
    import os  # 在函数开始时导入os模块
    
    # 生成缓存键
    cache_key = f"{knowledge_base_id}:{embedding_model_id}"
    
    # 检查缓存
    with _VECTOR_INDEX_CACHE_LOCK:
        if cache_key in _VECTOR_INDEX_CACHE:
            index, _ = _VECTOR_INDEX_CACHE[cache_key]
            # 更新最后访问时间
            _VECTOR_INDEX_CACHE[cache_key] = (index, time.time())
            logging.info(f"从缓存中加载向量索引: {cache_key}")
            return index
    
    # 缓存未命中，从磁盘加载
    try:
        project_root = Path(__file__).parent
        meta_config_path = project_root / 'meta' / knowledge_base_id / 'config.json'
        logging.info(f"知识库配置文件{meta_config_path}")
        if not meta_config_path.exists():
            raise ValueError(f"{knowledge_base_id}知识库配置文件不存在")
        with open(meta_config_path, encoding="utf-8") as f:
            config = json.load(f)
            vector_path = config.get("vector_path")
            if not vector_path:
                raise ValueError(f"{knowledge_base_id}知识库向量文件路径配置不存在")
            vector_path = os.path.abspath(os.path.join(project_root, vector_path))
        logging.info(f"知识库向量文件路径{vector_path}")
        if not os.path.exists(vector_path):
            raise ValueError(f"向量存储目录不存在: {vector_path}")
        required_files = ['default__vector_store.json', 'docstore.json', 'index_store.json']
        for file in required_files:
            if not os.path.exists(os.path.join(vector_path, file)):
                raise ValueError(f"缺少必要的向量文件: {file}")
        # 设置环境变量以避免OpenAI相关错误
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
        # 加载嵌入模型
        embedding_model = _initialize_embedding_model(embedding_model_id)
        # 保证分块方法和向量化一致
        text_splitter = LangchainNodeParser(
            SpacyTextSplitter(
                pipeline="zh_core_web_sm",
                chunk_size=1000,
                chunk_overlap=200
            )
        )
        from llama_index import ServiceContext
        service_context = ServiceContext.from_defaults(
            embed_model=embedding_model,
            text_splitter=text_splitter,
            llm=None  # 明确设置为None，避免尝试加载OpenAI模型
        )
        # 加载向量存储
        storage_context = StorageContext.from_defaults(
            persist_dir=vector_path,
            vector_store=FaissVectorStore.from_persist_dir(vector_path)
        )
        # 使用自定义的ServiceContext加载索引
        index = load_index_from_storage(
            storage_context=storage_context,
            service_context=service_context
        )
        # 将加载的索引放入缓存
        with _VECTOR_INDEX_CACHE_LOCK:
            _VECTOR_INDEX_CACHE[cache_key] = (index, time.time())
            # 检查是否需要清理缓存
            _cache_cleanup()
        logging.info(f"成功加载向量索引并添加到缓存: {cache_key}")
        return index
    except Exception as e:
        logging.error(f"加载向量索引失败: {str(e)}", exc_info=True)
        raise ValueError(f"加载知识库失败: {str(e)}")

def process_single_file(kb_id:str, org_file: str, vec_dir: str, embedding_model_id: str, incremental: bool, max_retries: int = 3) -> bool:
    """处理单个文件并生成向量"""
    def split_large_file(file_path: str, chunk_size: int = 500000) -> tuple:
        """将大文件分割成多个小文件"""
        temp_dir = tempfile.mkdtemp()
        chunks = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 按字符数分割
        for i in range(0, len(content), chunk_size):
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.txt")
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(content[i:i+chunk_size])
            chunks.append(chunk_path)
            
        return chunks, temp_dir

    # 确保向量存储目录存在且可写
    vec_dir = os.path.abspath(vec_dir)
    os.makedirs(vec_dir, exist_ok=True)
    # 第一次运行时强制关闭增量模式
    if not os.path.exists(os.path.join(vec_dir, "default__vector_store.json")):
        incremental = False

    for attempt in range(max_retries):
        try:
            # 确保向量存储目录存在
            os.makedirs(vec_dir, exist_ok=True)
            
            # 如果是目录则直接处理
            if os.path.isdir(org_file):
                process_documents(org_file, vec_dir, embedding_model_id, incremental=incremental, kb_id=kb_id)
                return True
                
            # 检查文件大小
            file_size = os.path.getsize(org_file)
            logging.info(f"文件大小: {file_size} 字节")
            logging.info(f"嵌入模型: {embedding_model_id} ")
            # 只用分块器分块，不再手动分割大文件
            # 创建临时目录结构
            temp_dir = os.path.join(vec_dir, "temp")
            # 清空临时目录，确保无历史残留
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            # 复制文件到临时目录
            temp_file = os.path.join(temp_dir, os.path.basename(org_file))
            shutil.copy2(org_file, temp_file)
            
            # 处理文档
            process_documents(
                temp_dir, 
                vec_dir, 
                embedding_model_id, 
                incremental=incremental,
                kb_id=kb_id
            )
            
            # 清理临时目录
            shutil.rmtree(temp_dir)
            return True
                
        except Exception as e:
            logging.error(f"处理文件 {org_file} 失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise ValueError(f"处理文件失败: {str(e)}")
            continue
            
    return False

def create_rag_engine(vector_index, chat_model_id, kb_id=None, embedding_model_id=None):
    """创建RAG引擎，根据模型ID选择合适的LLM，优先从缓存中读取"""
    try:
        # 如果提供了kb_id和embedding_model_id，则使用缓存
        if kb_id and embedding_model_id:
            cache_key = f"{kb_id}:{embedding_model_id}:{chat_model_id}"
            
            # 检查缓存
            with _RAG_ENGINE_CACHE_LOCK:
                if cache_key in _RAG_ENGINE_CACHE:
                    engine, _ = _RAG_ENGINE_CACHE[cache_key]
                    # 更新最后访问时间
                    _RAG_ENGINE_CACHE[cache_key] = (engine, time.time())
                    logging.info(f"从缓存中加载RAG引擎: {cache_key}")
                    return engine
        
        logging.info(f"正在创建RAG引擎，使用模型ID: {chat_model_id}")
        
        # 获取模型配置
        model_config = MODEL_CONFIG.get(chat_model_id)
        if not model_config:
            logging.warning(f"未找到模型配置: {chat_model_id}，使用MockLLM")
            from llama_index.llms import MockLLM
            llm = MockLLM()
        else:
            logging.info(f"使用模型配置: {json.dumps(model_config, ensure_ascii=False)}")
            
            # 根据模型类型选择不同的LLM实现
            if model_config.get("provider", "").upper() == "NVIDIA":
                # 使用NVIDIA模型
                logging.info("检测到NVIDIA模型，使用OpenAILike")
                from llama_index.llms.openai_like import OpenAILike
                
                # 直接使用OpenAILike，不需要自定义包装器
                llm = OpenAILike(
                    api_base=model_config.get('url'),
                    api_key=model_config.get('api_key'),
                    model=model_config.get('model'),
                    is_chat_model=True,
                    temperature=0.3,
                    max_tokens=1024,
                    streaming=True  # 启用流式输出
                )
                
                # 设置环境变量以避免OpenAI相关错误
                import os
                os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
                os.environ["OPENAI_API_BASE"] = model_config.get('url')
                
                logging.info(f"成功创建NVIDIA模型: {model_config.get('model')}")
            else:
                # 默认使用MockLLM
                logging.warning(f"不支持的模型提供商: {model_config.get('provider')}，使用MockLLM")
                from llama_index.llms import MockLLM
                llm = MockLLM()
        
        # 创建自定义提示模板，包含思考过程和答案分步的提示模板
        from llama_index.prompts import PromptTemplate

        # 定义带思考过程和答案分步的提示模板
        qa_template = PromptTemplate(
            """以下是关于用户问题的上下文信息:
{context_str}

请分两步回答：
1. 思考过程：请详细分析资料与问题的关系，推理得出答案。
2. 答案：用一句话直接回答用户问题。

问题：{query_str}
"""
        )
        
        # 创建查询引擎，配置为返回思考过程和引用源
        query_engine = vector_index.as_query_engine(
            llm=llm,
            similarity_top_k=3,
            response_mode="refine",  # 使用refine模式以获得更好的答案
            text_qa_template=qa_template,
            streaming=True,  # 启用流式输出
            verbose=True     # 显示详细信息
        )
        
        # 包装查询引擎，增加思考过程和引用源输出
        original_query = query_engine.query
        
        def enhanced_query(query_str, **kwargs):
            """增强查询函数，添加思考过程和引用源"""
            logging.info(f"开始查询: {query_str}")
            
            # 执行原始查询
            response = original_query(query_str, **kwargs)
            
            # 获取相关文档
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
            
            # 提取引用源信息
            sources = []
            for idx, node in enumerate(source_nodes):
                source_info = {
                    "index": idx,
                    "score": float(node.score) if hasattr(node, 'score') else None,
                    "file": node.metadata.get('file_path', '未知来源') if hasattr(node, 'metadata') else '未知来源',
                    "text": node.text[:200] + "..." if hasattr(node, 'text') and len(node.text) > 200 else (node.text if hasattr(node, 'text') else '未知内容')
                }
                sources.append(source_info)
                logging.info(f"引用源 {idx}: {source_info['file']}, 分数: {source_info['score']}")
            
            # 修改响应生成器，以包含思考过程和引用源
            if hasattr(response, 'response_gen'):
                original_gen = response.response_gen
                
                def enhanced_gen():
                    # 思考过程标记
                    thinking_sent = False
                    sources_sent = False
                    
                    # 思考过程内容
                    thinking_content = ""
                    in_thinking = False
                    
                    # 最终答案内容
                    answer_content = ""
                    
                    for chunk in original_gen:
                        # 检查是否包含思考过程标记
                        if "<思考过程>" in chunk and not thinking_sent:
                            in_thinking = True
                            thinking_content = chunk[chunk.find("<思考过程>") + 12:]
                            continue
                        if in_thinking and "</思考过程>" in chunk:
                            in_thinking = False
                            thinking_content += chunk[:chunk.find("</思考过程>")]
                            # 发送思考过程
                            yield json.dumps({
                                "type": "thinking",
                                "content": thinking_content.strip()
                            })
                            thinking_sent = True
                            # 处理思考后的内容
                            remaining = chunk[chunk.find("</思考过程>") + 13:]
                            if remaining.strip():
                                answer_content += remaining
                                yield json.dumps({
                                    "type": "answer",
                                    "content": remaining
                                })
                            continue
                        if in_thinking:
                            thinking_content += chunk
                            continue
                        # 正常处理答案部分
                        answer_content += chunk
                        yield json.dumps({
                            "type": "answer",
                            "content": chunk
                        })
                    
                    # 如果没有发送过思考内容，则发送默认思考过程
                    if not thinking_sent:
                        yield json.dumps({
                            "type": "thinking",
                            "content": "我正在分析相关知识，寻找问题的答案..."
                        })

                    # 如果答案内容为空，发送一个从源文档提取的答案
                    if not answer_content.strip():
                        extracted_answer = ""
                        for source in sources:
                            if "A：" in source["text"] or "A:" in source["text"]:
                                answer_text = source["text"]
                                if "A：" in answer_text:
                                    extracted_answer = answer_text.split("A：")[1].strip()
                                elif "A:" in answer_text:
                                    extracted_answer = answer_text.split("A:")[1].strip()
                                if extracted_answer:
                                    yield json.dumps({
                                        "type": "answer",
                                        "content": extracted_answer
                                    })
                                    break
                        if not extracted_answer:
                            # 如果没有提取到答案，直接返回检索内容
                            if sources and "text" in sources[0]:
                                yield json.dumps({
                                    "type": "answer",
                                    "content": sources[0]["text"]
                                })
                            else:
                                yield json.dumps({
                                    "type": "answer",
                                    "content": "根据知识库内容，我找到了相关信息但无法给出确切答案。请查看引用内容获取更多信息。"
                                })
                    
                    # 发送引用源
                    if sources:
                        yield json.dumps({
                            "type": "sources",
                            "content": sources
                        })
                    else:
                        yield json.dumps({
                            "type": "sources",
                            "content": [{"index": 0, "file": "知识库", "text": "未找到相关内容"}]
                        })
                
                # 替换原始生成器
                response.response_gen = enhanced_gen()
            
            return response
        
        # 替换查询函数
        query_engine.query = enhanced_query
        
        # 如果提供了kb_id和embedding_model_id，则缓存RAG引擎
        if kb_id and embedding_model_id:
            cache_key = f"{kb_id}:{embedding_model_id}:{chat_model_id}"
            with _RAG_ENGINE_CACHE_LOCK:
                _RAG_ENGINE_CACHE[cache_key] = (query_engine, time.time())
                # 检查是否需要清理缓存
                _cache_cleanup()
            logging.info(f"成功创建RAG引擎并添加到缓存: {cache_key}")
        
        logging.info("RAG引擎创建成功")
        return query_engine
    except Exception as e:
        logging.error(f"创建RAG引擎失败: {str(e)}", exc_info=True)
        raise ValueError(f"创建RAG引擎失败: {str(e)}")

def _cache_cleanup():
    """清理过期的缓存项"""
    # 清理向量索引缓存
    with _VECTOR_INDEX_CACHE_LOCK:
        if len(_VECTOR_INDEX_CACHE) > _VECTOR_INDEX_CACHE_MAX_SIZE:
            # 按最后访问时间排序
            sorted_cache = sorted(_VECTOR_INDEX_CACHE.items(), key=lambda x: x[1][1])
            # 删除最旧的项
            for key, _ in sorted_cache[:len(_VECTOR_INDEX_CACHE) - _VECTOR_INDEX_CACHE_MAX_SIZE]:
                logging.info(f"从缓存中删除向量索引: {key}")
                del _VECTOR_INDEX_CACHE[key]
    
    # 清理RAG引擎缓存
    with _RAG_ENGINE_CACHE_LOCK:
        if len(_RAG_ENGINE_CACHE) > _RAG_ENGINE_CACHE_MAX_SIZE:
            # 按最后访问时间排序
            sorted_cache = sorted(_RAG_ENGINE_CACHE.items(), key=lambda x: x[1][1])
            # 删除最旧的项
            for key, _ in sorted_cache[:len(_RAG_ENGINE_CACHE) - _RAG_ENGINE_CACHE_MAX_SIZE]:
                logging.info(f"从缓存中删除RAG引擎: {key}")
                del _RAG_ENGINE_CACHE[key]