
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentMeta(BaseModel):
    """单个文档元数据"""
    doc_id: str  # 文档ID（文件名）
    name: str  # 显示名称
    status: str = "未处理"  # 状态：未处理/处理成功/处理失败
    create_time: datetime = datetime.now()
    update_time: datetime = datetime.now()

class KnowledgeBaseMeta(BaseModel):
    """子知识库元数据"""
    kb_id: str  # 知识库ID
    source_path: str  # 源文件路径
    vector_path: str  # 向量存储路径
    embedding_model: str  # 使用的embedding模型
    incremental: bool = True  # 是否增量更新
    status: str = "启用"  # 状态：启用/停用 
    documents: Dict[str, DocumentMeta] = {}  # 文档元数据字典
    create_time: datetime = datetime.now()
    update_time: datetime = datetime.now()

class KnowledgeMetaManager:
    """知识库元数据管理器"""
    
    def __init__(self, meta_path="knowledge_meta.json"):
        self.meta_path = meta_path
        self.knowledge_bases: Dict[str, KnowledgeBaseMeta] = {}
        self.lock = threading.Lock()
        self.load_meta()
    
    def load_meta(self):
        """加载元数据（线程安全）"""
        with self.lock:
            try:
                if os.path.exists(self.meta_path):
                    # 创建备份
                    backup_path = f"{self.meta_path}.bak"
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(self.meta_path, backup_path)
                    
                    with open(backup_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.knowledge_bases = {
                            kb_id: KnowledgeBaseMeta(**meta) 
                            for kb_id, meta in data.items()
                        }
                    logger.info(f"成功加载元数据，共 {len(self.knowledge_bases)} 个知识库")
            except Exception as e:
                logger.error(f"加载元数据失败: {str(e)}")
                raise RuntimeError("加载元数据失败，请检查文件完整性")
    
    def save_meta(self):
        """保存元数据（线程安全）"""
        with self.lock:
            try:
                # 先写入临时文件
                temp_path = f"{self.meta_path}.tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump({
                        kb_id: kb.dict() 
                        for kb_id, kb in self.knowledge_bases.items()
                    }, f, ensure_ascii=False, indent=2, default=str)
                
                # 原子性替换原文件
                if os.path.exists(self.meta_path):
                    os.replace(temp_path, self.meta_path)
                else:
                    os.rename(temp_path, self.meta_path)
                
                logger.info("元数据保存成功")
            except Exception as e:
                logger.error(f"保存元数据失败: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise RuntimeError("保存元数据失败，请重试")
    
    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBaseMeta]:
        """获取知识库元数据"""
        return self.knowledge_bases.get(kb_id)
    
    def update_knowledge_base(self, kb_meta: KnowledgeBaseMeta):
        """更新知识库元数据"""
        self.knowledge_bases[kb_meta.kb_id] = kb_meta
        self.save_meta()
    
    def list_knowledge_bases(self) -> List[KnowledgeBaseMeta]:
        """列出所有知识库"""
        return list(self.knowledge_bases.values())
    
    def count_knowledge_bases(self) -> int:
        """统计知识库数量"""
        return len(self.knowledge_bases)
