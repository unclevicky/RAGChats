import os
import sys
import json
import logging
from pathlib import Path
from pathlib import Path
import requests
import shutil
import tempfile
#from backend.config import MODEL_CONFIG
import json
import faiss
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings,SimpleDirectoryReader,VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain_text_splitters import SpacyTextSplitter


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

def load_model_config():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载模型配置失败: {str(e)}")
        return {}

MODEL_CONFIG = load_model_config()


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

class SiliconFlowEmbedding(BaseEmbedding):
    """硅基流动API的嵌入模型实现
    
    属性:
        api_url (str): API端点URL
        api_key (str): 认证密钥
        model_name (str): 模型名称
    """
    
    def __init__(self, api_url: str, api_key: str, model_name: str, **kwargs):
        """初始化硅基流动嵌入模型
        
        Args:
            api_url: API端点URL
            api_key: 认证密钥
            model_name: 模型名称
            **kwargs: 传递给父类的额外参数
            
        Raises:
            ValueError: 如果缺少必需参数
        """
        super().__init__(**kwargs)
        self._api_url = api_url
        self._api_key = api_key
        self._model_name = model_name
        
        if not all([self._api_url, self._api_key, self._model_name]):
            raise ValueError("API URL、API Key和Model Name都是必需参数")

    @property
    def api_url(self) -> str:
        """获取API端点URL"""
        return self._api_url

    @property
    def api_key(self) -> str:
        """获取认证密钥"""
        return self._api_key

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self._model_name

    def _get_query_embedding(self, query: str) -> list[float]:
        """获取查询文本的嵌入向量"""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """异步获取查询文本的嵌入向量"""
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """获取单个文本的嵌入向量
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            文本的嵌入向量
            
        Raises:
            ConnectionError: 如果API请求失败
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._model_name,
            "input": text
        }
        
        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API请求失败: {str(e)}")

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本的嵌入向量
        
        Args:
            texts: 要嵌入的文本列表
            
        Returns:
            嵌入向量列表
        """
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """异步获取单个文本的嵌入向量"""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """异步批量获取文本的嵌入向量"""
        return [await self._aget_text_embedding(text) for text in texts]

from backend.knowledge_meta import KnowledgeMetaManager, KnowledgeBaseMeta, DocumentMeta
from datetime import datetime

meta_manager = KnowledgeMetaManager()

def process_documents(file_path, vector_path, embedding_model_id, incremental=True, kb_id=None):
    """处理文档并创建/更新向量索引
    
    Args:
        file_path: 文档路径
        vector_path: 向量存储路径
        embedding_model_id: 嵌入模型ID
        incremental: 是否增量更新
        kb_id: 知识库ID（用于元数据跟踪）
        
    Returns:
        tuple: (更新后的向量索引, 处理结果列表)
        
    Raises:
        ValueError: 如果处理失败
    """
        
    # 确保路径存在
    os.makedirs(vector_path, exist_ok=True)
    os.makedirs(file_path, exist_ok=True)
    
    logging.info(f"1.确认知识库 {kb_id}，原始文件路径: {file_path}，向量存储路径: {vector_path}, 嵌入模型: {embedding_model_id}")

        
    # 确保路径存在
    os.makedirs(vector_path, exist_ok=True)
    os.makedirs(file_path, exist_ok=True)
    
    logging.info(f"1.确认知识库 {kb_id}，原始文件路径: {file_path}，向量存储路径: {vector_path}, 嵌入模型: {embedding_model_id}")

    # 获取或创建知识库元数据
    kb_meta = meta_manager.get_knowledge_base(kb_id) if kb_id else None
    if not kb_meta:
        kb_meta = KnowledgeBaseMeta(
            kb_id=kb_id or os.path.basename(vector_path),
            source_path=file_path,
            vector_path=vector_path,
            embedding_model=embedding_model_id,
            incremental=incremental
        )
    
    # 加载文档并更新元数据
    loader = SimpleDirectoryReader(file_path)
    documents = loader.load_data()
    results = []
    
    # 更新文档元数据状态
    for doc in documents:
        doc_id = os.path.basename(doc.metadata.get('file_path', ''))
        if doc_id:
            doc_meta = kb_meta.documents.get(doc_id, DocumentMeta(doc_id=doc_id, name=doc_id))
            doc_meta.status = "处理中"
            kb_meta.documents[doc_id] = doc_meta
    
    try:
        # 根据embedding_model_id初始化模型
        embedding_config = MODEL_CONFIG.get(embedding_model_id)
        if not embedding_config:
            raise ValueError(f"未找到模型配置: {embedding_model_id}")
        
        if embedding_model_id == "bge-large-zh-v1.5":
            # 验证必需配置项
            required_keys = ['url', 'api_key', 'model']
            if not all(key in embedding_config for key in required_keys):
                raise ValueError(f"硅基流动模型配置缺少必需参数，需要: {required_keys}")
                
            try:
                embedding_model = SiliconFlowEmbedding(
                    api_url=str(embedding_config['url']),
                    api_key=str(embedding_config['api_key']),
                    model_name=str(embedding_config['model'])
                )
            except Exception as e:
                error_msg = f"创建SiliconFlowEmbedding实例失败: {str(e)}. 配置参数: {embedding_config}"
                logging.error(error_msg)
                raise ValueError(error_msg)
        elif embedding_model_id.startswith("huggingface_"):
            if 'model' not in embedding_config:
                raise ValueError("HuggingFace模型配置缺少model参数")
                
            # 优先尝试本地模型路径
            model_name = str(embedding_config['model'])
            # 此写法不兼容打包后的exe文件模式
            # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local_model_path = Path(project_root) / "model_cache" / model_name
            logging.info(f"尝试使用本地模型路径: {local_model_path}")
            # 检查本地缓存路径是否存在
            if not local_model_path.exists():
                # 尝试系统默认缓存路径
                default_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
                if default_cache.exists():
                    # 找到最新的snapshot
                    snapshots = list(default_cache.glob("snapshots/*"))
                    if snapshots:
                        local_model_path = snapshots[-1]  # 使用最新的snapshot
            
            if local_model_path.exists():
                logging.info(f"使用本地模型路径: {local_model_path}")
                embedding_model = HuggingFaceEmbedding(
                    model_name=str(local_model_path),
                    device="cpu"  # 明确指定设备
                )
            else:
                logging.warning(f"本地模型未找到，尝试在线下载: {model_name}")
                embedding_model = HuggingFaceEmbedding(
                    model_name=model_name,
                    device="cpu",  # 明确指定设备
                    trust_remote_code=True
                )
        else:
            raise ValueError(f"不支持的嵌入模型ID: {embedding_model_id}")

        Settings.embed_model = embedding_model
        Settings.text_splitter = LangchainNodeParser(SpacyTextSplitter(
            pipeline="zh_core_web_sm",
            chunk_size=1000,
            chunk_overlap=200
        ))

        logging.info(f"2.初始化完成,处理文件 {file_path}，向量存储路径 {vector_path}，嵌入模型ID {embedding_model_id}，增量更新 {incremental}，知识库ID {kb_id}")
        
        # 确保向量存储目录存在并可写
        try:
            os.makedirs(vector_path, exist_ok=True)
            # 测试目录可写性
            test_file = os.path.join(vector_path, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            error_msg = f"无法写入向量存储目录 {vector_path}: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        if incremental and os.path.exists(os.path.join(vector_path, "default__vector_store.json")):
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=vector_path,
                    vector_store=FaissVectorStore.from_persist_dir(vector_path)
                )
                index = load_index_from_storage(storage_context)
                for doc in documents:
                    index.insert(doc)
            except Exception as e:
                logging.error(f"加载已有向量存储失败，将创建新索引: {str(e)}")
                dimension = 1024
                faiss_index = faiss.IndexFlatL2(dimension)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        logging.info(f"2.初始化完成,处理文件 {file_path}，向量存储路径 {vector_path}，嵌入模型ID {embedding_model_id}，增量更新 {incremental}，知识库ID {kb_id}")
        
        # 确保向量存储目录存在并可写
        try:
            os.makedirs(vector_path, exist_ok=True)
            # 测试目录可写性
            test_file = os.path.join(vector_path, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            error_msg = f"无法写入向量存储目录 {vector_path}: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        if incremental and os.path.exists(os.path.join(vector_path, "default__vector_store.json")):
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=vector_path,
                    vector_store=FaissVectorStore.from_persist_dir(vector_path)
                )
                index = load_index_from_storage(storage_context)
                for doc in documents:
                    index.insert(doc)
            except Exception as e:
                logging.error(f"加载已有向量存储失败，将创建新索引: {str(e)}")
                dimension = 1024
                faiss_index = faiss.IndexFlatL2(dimension)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        else:
            # 确保目录存在且可写
            os.makedirs(vector_path, exist_ok=True)
            # 确保目录存在且可写
            os.makedirs(vector_path, exist_ok=True)
            dimension = 1024
            faiss_index = faiss.IndexFlatL2(dimension)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        logging.info(f"3.向量化处理完成,处理文件 {file_path}，向量存储路径 {vector_path}，嵌入模型ID {embedding_model_id}，增量更新 {incremental}，知识库ID {kb_id}")
        
        
        # 在持久化前添加路径验证
        if not os.path.exists(vector_path):
            os.makedirs(vector_path, exist_ok=True)

        # 持久化前再次检查目录可写性
        try:
            test_file = os.path.join(vector_path, ".write_test")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            error_msg = f"无法写入向量存储目录 {vector_path}: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.info(f"3.向量化处理完成,处理文件 {file_path}，向量存储路径 {vector_path}，嵌入模型ID {embedding_model_id}，增量更新 {incremental}，知识库ID {kb_id}")
        
        
        # 在持久化前添加路径验证
        if not os.path.exists(vector_path):
            os.makedirs(vector_path, exist_ok=True)

        # 持久化前再次检查目录可写性
        try:
            test_file = os.path.join(vector_path, ".write_test")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            error_msg = f"无法写入向量存储目录 {vector_path}: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # 存储更新后的向量
        try:
            # 确保使用绝对路径
            persist_path = os.path.abspath(vector_path)
            # 确保路径使用正斜杠
            persist_path = persist_path.replace('\\', '/')
            index.storage_context.persist(persist_dir=persist_path)
            # 确保路径使用正斜杠
            persist_path = persist_path.replace('\\', '/')
            # 确保目录存在
            os.makedirs(persist_path, exist_ok=True)
            # 测试目录可写性
            test_file = os.path.join(persist_path, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
            # 执行持久化
            persist_path=Path(persist_path)
            index.storage_context.persist(persist_dir=str(persist_path))
        except Exception as e:
            error_msg = f"持久化失败: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.info(f"4.持久化完成,处理文件 {file_path}，向量存储路径 {vector_path}，嵌入模型ID {embedding_model_id}，增量更新 {incremental}，知识库ID {kb_id}")
        try:
            # 确保使用绝对路径
            persist_path = os.path.abspath(vector_path)
            # 确保路径使用正斜杠
            persist_path = persist_path.replace('\\', '/')
            index.storage_context.persist(persist_dir=persist_path)
            # 确保路径使用正斜杠
            persist_path = persist_path.replace('\\', '/')
            # 确保目录存在
            os.makedirs(persist_path, exist_ok=True)
            # 测试目录可写性
            test_file = os.path.join(persist_path, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
            # 执行持久化
            persist_path=Path(persist_path)
            index.storage_context.persist(persist_dir=str(persist_path))
        except Exception as e:
            error_msg = f"持久化失败: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.info(f"4.持久化完成,处理文件 {file_path}，向量存储路径 {vector_path}，嵌入模型ID {embedding_model_id}，增量更新 {incremental}，知识库ID {kb_id}")
        
        # 处理成功后更新状态
        for doc in documents:
            doc_id = os.path.basename(doc.metadata.get('file_path', ''))
            if doc_id in kb_meta.documents:
                kb_meta.documents[doc_id].status = "处理成功"
                kb_meta.documents[doc_id].update_time = datetime.now()
                results.append({
                    "doc_id": doc_id,
                    "status": "success",
                    "message": "处理成功"
                })
        
        kb_meta.status = "启用"
        kb_meta.update_time = datetime.now()
        meta_manager.update_knowledge_base(kb_meta)
        
        logging.info(f"5.元数据更新完成,处理文件 {file_path}，向量存储路径 {vector_path}，嵌入模型ID {embedding_model_id}，增量更新 {incremental}，知识库ID {kb_id}")

        
        logging.info(f"5.元数据更新完成,处理文件 {file_path}，向量存储路径 {vector_path}，嵌入模型ID {embedding_model_id}，增量更新 {incremental}，知识库ID {kb_id}")

        return index, results
        
    except Exception as e:
        # 处理失败更新状态
        for doc in documents:
            doc_id = os.path.basename(doc.metadata.get('file_path', ''))
            if doc_id in kb_meta.documents:
                kb_meta.documents[doc_id].status = "处理失败"
                kb_meta.documents[doc_id].update_time = datetime.now()
                results.append({
                    "doc_id": doc_id,
                    "status": "failed",
                    "message": str(e)
                })
        
        kb_meta.update_time = datetime.now()
        meta_manager.update_knowledge_base(kb_meta)
        raise ValueError(f"处理失败: {str(e)}")

def load_vector_index(knowledge_base_id, embedding_model_id: str = "bge-large-zh-v1.5"):
    # 获取项目根目录(假设utils.py在backend目录下)
    # 此模式不兼容打包后的exe文件模式
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 构建绝对路径
    # vector_path = os.path.normpath(os.path.join(project_root, 'backend', 'vectorstore', knowledge_base_id))
    # 从元数据获取向量存储路径
    meta_config_path = Path(project_root) / 'meta' / knowledge_base_id / 'config.json'
    logging.info(f"知识库配置文件{meta_config_path}")
    if not meta_config_path.exists():
        raise ValueError(f"{knowledge_base_id}知识库配置文件不存在")

    
    with open(meta_config_path, encoding="utf-8") as f:
        config = json.load(f)
        vector_path = config.get("vector_path")
        if not vector_path:
            raise ValueError(f"{knowledge_base_id}知识库向量文件路径配置不存在")
    
    # 确保使用绝对路径
    vector_path = os.path.join(project_root, vector_path)
    vector_path = os.path.abspath(vector_path)
    logging.info(f"知识库向量文件路径{vector_path}")
    
    # 检查向量存储目录是否存在
    if not os.path.exists(vector_path):
        raise ValueError(f"向量存储目录不存在: {vector_path}")
    
    # 检查必要的向量文件是否存在
    required_files = ['default__vector_store.json', 'docstore.json', 'index_store.json']
    for file in required_files:
        if not os.path.exists(os.path.join(vector_path, file)):
            raise ValueError(f"缺少必要的向量文件: {file}")

    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=vector_path,
            vector_store=FaissVectorStore.from_persist_dir(vector_path)
        )

        # 根据embedding_model_id初始化模型
        embedding_config = MODEL_CONFIG.get(embedding_model_id)
        if not embedding_config:
            raise ValueError(f"未找到模型配置: {embedding_model_id}")
        
        if embedding_model_id == "bge-large-zh-v1.5":
            required_keys = ['url', 'api_key', 'model']
            if not all(key in embedding_config for key in required_keys):
                raise ValueError(f"硅基流动模型配置缺少必需参数，需要: {required_keys}")
                
            embedding_model = SiliconFlowEmbedding(
                api_url=str(embedding_config['url']),
                api_key=str(embedding_config['api_key']),
                model_name=str(embedding_config['model'])
            )
        elif embedding_model_id.startswith("huggingface_"):
            if 'model' not in embedding_config:
                raise ValueError("HuggingFace模型配置缺少model参数")

            # 优先尝试本地模型路径
            model_name = str(embedding_config['model'])
            local_model_path =  Path(project_root)  / "model_cache" / model_name
            
            logging.info(f"尝试使用本地模型路径: {local_model_path}")
            # 检查本地缓存路径是否存在
            if not local_model_path.exists():
                # 尝试系统默认缓存路径
                default_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
                if default_cache.exists():
                    # 找到最新的snapshot
                    snapshots = list(default_cache.glob("snapshots/*"))
                    if snapshots:
                        local_model_path = snapshots[-1]  # 使用最新的snapshot
            
            if local_model_path.exists():
                logging.info(f"使用本地模型路径: {local_model_path}")
                embedding_model = HuggingFaceEmbedding(
                    model_name=str(local_model_path),
                    device="cpu"  # 明确指定设备
                )
            else:
                logging.warning(f"本地模型未找到，尝试在线下载: {model_name}")
                embedding_model = HuggingFaceEmbedding(
                    model_name=model_name,
                    device="cpu",  # 明确指定设备
                    trust_remote_code=True
                )
        else:
            raise ValueError(f"不支持的嵌入模型ID: {embedding_model_id}")

        Settings.embed_model = embedding_model
        Settings.text_splitter = LangchainNodeParser(SpacyTextSplitter(
            pipeline="zh_core_web_sm",
            chunk_size=1000,
            chunk_overlap=200
        ))

        index = load_index_from_storage(storage_context, embed_model=Settings.embed_model)
        return index
        
    except Exception as e:
        logging.error(f"加载向量索引失败: {str(e)}", exc_info=True)
        raise ValueError(f"加载知识库失败: {str(e)}")

def process_single_file(kb_id:str, org_file: str, vec_dir: str, embedding_model_id: str, incremental: bool, max_retries: int = 3) -> bool:
    """处理单个文件并生成向量
    
    Args:
        org_file: 原始文件路径
        vec_dir: 向量存储目录
        embedding_model_id: 嵌入模型配置ID，默认为"bge-large-zh-v1.5"
        incremental: 是否增量处理
        max_retries: 最大重试次数
        max_retries: 最大重试次数
        
    Returns:
        bool: 处理是否成功
    """
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

    # 在调用 process_documents 前处理路径
    # 针对Windows中文路径问题
    # vec_dir = vec_dir.encode('utf-8').decode('gbk') 
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
            if file_size > 500000 and not embedding_model_id.startswith("huggingface_"):  # 大于500KB的文件需要分割
                chunks, temp_dir = split_large_file(org_file)
                try:
                    for chunk in chunks:
                        # 处理每个分块
                        process_documents(
                            os.path.dirname(chunk),
                            vec_dir,
                            embedding_model_id,
                            incremental=incremental,
                            kb_id=os.path.basename(os.path.dirname(vec_dir))
                        )
                    return True
                finally:
                    # 清理临时文件
                    shutil.rmtree(temp_dir)
            else:
                # 创建临时目录结构
                temp_dir = os.path.join(vec_dir, "temp")
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
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 413 and attempt < max_retries - 1:
                # 如果是413错误且还有重试机会，减小分块大小
                logging.warning(f"请求过大，尝试减小分块大小 (重试 {attempt + 1}/{max_retries})")
                continue
            raise
        except Exception as e:
            logging.error(f"处理文件 {org_file} 失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise ValueError(f"处理文件失败: {str(e)}")
            continue
            
    return False
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

    # 在调用 process_documents 前处理路径
    # 针对Windows中文路径问题
    # vec_dir = vec_dir.encode('utf-8').decode('gbk') 
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
            if file_size > 500000 and not embedding_model_id.startswith("huggingface_"):  # 大于500KB的文件需要分割
                chunks, temp_dir = split_large_file(org_file)
                try:
                    for chunk in chunks:
                        # 处理每个分块
                        process_documents(
                            os.path.dirname(chunk),
                            vec_dir,
                            embedding_model_id,
                            incremental=incremental,
                            kb_id=os.path.basename(os.path.dirname(vec_dir))
                        )
                    return True
                finally:
                    # 清理临时文件
                    shutil.rmtree(temp_dir)
            else:
                # 创建临时目录结构
                temp_dir = os.path.join(vec_dir, "temp")
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
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 413 and attempt < max_retries - 1:
                # 如果是413错误且还有重试机会，减小分块大小
                logging.warning(f"请求过大，尝试减小分块大小 (重试 {attempt + 1}/{max_retries})")
                continue
            raise
        except Exception as e:
            logging.error(f"处理文件 {org_file} 失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise ValueError(f"处理文件失败: {str(e)}")
            continue
            
    return False

def create_rag_engine(vector_index, chat_model_id):
    try:
        # 验证模型配置存在
        if chat_model_id not in MODEL_CONFIG:
            raise ValueError(f"未找到模型配置: {chat_model_id}")
            
        config = MODEL_CONFIG[chat_model_id]
        llm = OpenAILike(
            api_base=config['url'],
            api_key=config['api_key'],
            model=config['model'],
            is_chat_model=True
        )
        # 创建支持流式输出的查询引擎
        query_engine = vector_index.as_query_engine(llm=llm, streaming=True)
        
        return query_engine
    except Exception as e:
        logging.error(f"创建RAG引擎失败: {str(e)}", exc_info=True)
        raise ValueError(f"创建RAG引擎失败: {str(e)}")
    try:
        # 验证模型配置存在
        if chat_model_id not in MODEL_CONFIG:
            raise ValueError(f"未找到模型配置: {chat_model_id}")
            
        config = MODEL_CONFIG[chat_model_id]
        llm = OpenAILike(
            api_base=config['url'],
            api_key=config['api_key'],
            model=config['model'],
            is_chat_model=True
        )
        # 创建支持流式输出的查询引擎
        query_engine = vector_index.as_query_engine(llm=llm, streaming=True)
        
        return query_engine
    except Exception as e:
        logging.error(f"创建RAG引擎失败: {str(e)}", exc_info=True)
        raise ValueError(f"创建RAG引擎失败: {str(e)}")