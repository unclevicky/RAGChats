
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Body
import os
from datetime import datetime
import json
import re
import shutil
import urllib.parse
import tempfile
import logging
from pathlib import Path
from typing import List, Dict
from ..utils import process_single_file, process_documents
from pypinyin import pinyin, Style
import sys
from fastapi.middleware.cors import CORSMiddleware


def to_pinyin(text, capitalize=False):
    """将中文转换为拼音
    Args:
        text: 要转换的文本
        capitalize: 是否首字母大写(默认False)
    Returns:
        转换后的拼音字符串
    """
    result = []
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # 中文字符
            char_pinyin = pinyin(char, style=Style.NORMAL)[0][0]
            result.append(char_pinyin.capitalize() if capitalize else char_pinyin)
        else:
            result.append(char.capitalize() if capitalize else char)
    return ''.join(result)

from pypinyin import pinyin, Style

router = APIRouter(prefix="/knowledge-bases", tags=["KnowledgeBases"])
def get_project_root():
    if getattr(sys, 'frozen', False):
        # 打包环境：使用临时解压目录或EXE所在目录
        if hasattr(sys, '_MEIPASS'):
            return sys._MEIPASS  # 单文件模式临时目录
        else:
            return os.path.dirname(sys.executable)  # 单文件夹模式EXE目录
    else:
        # 开发环境：基于__file__计算路径
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 并不是project_root,而是backend的路径
project_root = get_project_root()
#这种写法不兼容打包后exe运行模式
#DATA_DIR = Path(__file__).parent.parent / "data"
#META_DIR = Path(__file__).parent.parent / "meta"
DATA_DIR = Path(project_root) / "data"
META_DIR = Path(project_root) / "meta"

@router.get("/")
async def list_knowledge_bases():
    """获取所有知识库列表"""
    knowledge_bases = []
    for kb in DATA_DIR.iterdir():
        if not kb.is_dir():
            continue
            
        kb_info = {
            "id": kb.name,
            "name": kb.name.replace("_", " ").title()
        }
        
        # 加载知识库状态信息
        meta_path = META_DIR / kb.name / "config.json"
        try:
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                    kb_info["status"] = meta.get("status", "active")
        except Exception as e:
            print(f"加载知识库 {kb.name} 元数据失败: {str(e)}")
            kb_info["status"] = "active"  # 默认状态
            
        knowledge_bases.append(kb_info)
    
    return knowledge_bases

@router.post("/")
async def create_knowledge_base(kb_id: str, kb_name: str = None):
    """创建新知识库"""
    # 优先使用kb_name参数作为知识库ID
    if kb_name:
        kb_id = kb_name.strip().replace(" ", "_").lower()
    
    # 验证知识库ID格式（支持汉字、字母、数字、下划线和连字符）
    if not kb_id or not re.match(r'^[\w\u4e00-\u9fa5-]+$', kb_id):
        raise HTTPException(status_code=400, detail="知识库ID只能包含汉字、字母、数字、下划线和连字符")
    
    # 创建目录结构
    kb_path = DATA_DIR / kb_id
    # 对vectorstore目录使用拼音命名(首字母大写)
    vec_kb_id = to_pinyin(kb_id, capitalize=True) if any(ord(c) > 127 for c in kb_id) else kb_id
    vec_path = DATA_DIR.parent / "vectorstore" / vec_kb_id

    
    if kb_path.exists():
        raise HTTPException(status_code=400, detail="Knowledge base already exists")
    
    kb_path.mkdir(parents=True, exist_ok=True)
    vec_path.mkdir(parents=True, exist_ok=True)
    
    # 创建元数据目录
    meta_kb_dir = META_DIR / kb_id
    meta_kb_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建知识库元数据配置文件
    meta_config_path = meta_kb_dir / "config.json"
    config = {
        "kb_id": kb_id,
        "embedding_model": "huggingface_bge-large-zh-v1.5",
        "incremental": "true",
        "original_kb_id": kb_id,  # 保留原始ID
        "vector_kb_id": vec_kb_id,  # 记录向量存储使用的ID
        "pinyin_conversion": True if any(ord(c) > 127 for c in kb_id) else False,  # 标记是否经过拼音转换
        "source_path": str(kb_path.relative_to(DATA_DIR.parent.parent)),
        "vector_path": str(vec_path.relative_to(DATA_DIR.parent)),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "status": "active"
    }
    
    with open(meta_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    return {
        "id": kb_id,
        "source_path": str(kb_path),
        "vector_path": str(vec_path),
        "status": "created",
        "config_path": str(meta_config_path)
    }

@router.get("/{kb_id}")
async def get_knowledge_base(kb_id: str):
    """获取单个知识库详情"""
    kb_path = DATA_DIR / kb_id
    if not kb_path.exists() or not kb_path.is_dir():
        raise HTTPException(status_code=404, detail="知识库不存在")
    
    meta_path = META_DIR / kb_id / "config.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}
    
    # 获取文件列表
    files = [f for f in kb_path.iterdir() if f.is_file()]
    
    return {
        "id": kb_id,
        "name": kb_id.replace("_", " ").title(),
        "source_path": str(meta.get("source_path")),
        "vector_path": str(meta.get("vector_path")),
        "file_count": len(files),
        "embedding_model": meta.get("embedding_model"),
        "incremental": meta.get("incremental"),
        "status": meta.get("status", "启用"),
        "created_at": meta.get("created_at", datetime.fromtimestamp(kb_path.stat().st_ctime).isoformat()),
        "updated_at": meta.get("updated_at", ""),
        "config_path": str(meta_path)
    }

@router.put("/{kb_id}")
async def update_knowledge_base(kb_id: str, payload: Dict):
    """更新知识库设置"""
    kb_path = DATA_DIR / kb_id
    if not kb_path.exists() or not kb_path.is_dir():
        raise HTTPException(status_code=404, detail="知识库不存在")
    
    # 元数据配置文件路径
    meta_config_path = META_DIR / kb_id / "config.json"
    
    # 读取现有配置或创建新配置
    config = {
        "embedding_model": "bge-large-zh-v1.5",
        "incremental": True,
        "status": "active",
        "updated_at": datetime.now().isoformat()
    }
    
    # 如果配置文件存在，加载现有配置
    if meta_config_path.exists():
        with open(meta_config_path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    
    # 更新配置
    config.update({
        "embedding_model": payload.get("embedding_model", config["embedding_model"]),
        "incremental": payload.get("incremental", config["incremental"]),
        "status": payload.get("status", config["status"]),
        "updated_at": datetime.now().isoformat()
    })
    
    # 保存配置
    with open(meta_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    return {
        "id": kb_id,
        "status": "updated",
        "embedding_model": config["embedding_model"],
        "incremental": config["incremental"],
        "status": config["status"]
    }

@router.delete("/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    """删除知识库及其所有相关数据"""

    # 从元数据获取向量存储路径
    meta_config_path = META_DIR / kb_id / "config.json"
    if not meta_config_path.exists():
        raise HTTPException(
            status_code=404,
            detail="知识库配置文件不存在"
        )
    
    with open(meta_config_path, encoding="utf-8") as f:
        config = json.load(f)
        vector_path = config.get("vector_path")
        if not vector_path:
            raise HTTPException(
                status_code=500,
                detail="配置中缺少向量存储路径"
            )


    # 定义要删除的目录
    dirs_to_delete = [
        DATA_DIR / kb_id,
        DATA_DIR.parent / vector_path,
        META_DIR / kb_id
    ]
    
    # 检查知识库是否存在
    if not dirs_to_delete[0].exists():
        raise HTTPException(status_code=404, detail="知识库不存在")
    
    try:
        # 删除所有相关目录
        for dir_path in dirs_to_delete:
            if dir_path.exists():
                shutil.rmtree(dir_path)
        
        return {
            "kb_id": kb_id,
            "status": "deleted",
            "message": "知识库及其所有数据已成功删除"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除知识库失败: {str(e)}"
        )

@router.post("/{kb_id}/files")
async def upload_file(
    kb_id: str, 
    file: UploadFile = File(..., description="上传的文件")
):
    """上传文件到知识库并保存元数据"""
    try:
        kb_path = DATA_DIR / kb_id
        if not kb_path.exists():
            raise HTTPException(status_code=404, detail="知识库不存在")
        
        # 确保文件名安全
        filename = file.filename.replace("/", "_").replace("\\", "_")
        file_path = kb_path / filename
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 创建文件元数据文件
        meta_kb_dir = META_DIR / kb_id
        meta_kb_dir.mkdir(parents=True, exist_ok=True)
        
        file_meta_path = meta_kb_dir / f"{filename}.json"
        file_metadata = {
            "name": filename,
            "size": file_path.stat().st_size,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "uploaded",
            "path": str(file_path.relative_to(DATA_DIR.parent.parent)),
            "kb_id": kb_id
        }
        
        with open(file_meta_path, "w", encoding="utf-8") as f:
            json.dump(file_metadata, f, ensure_ascii=False, indent=2)
        
        # 更新知识库元数据的更新时间
        meta_config_path = meta_kb_dir / "config.json"
        if meta_config_path.exists():
            with open(meta_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config["updated_at"] = datetime.now().isoformat()
            with open(meta_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        
        return {
            "filename": filename,
            "kb_id": kb_id,
            "metadata": file_metadata,
            "status": "uploaded"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"文件上传失败: {str(e)}"
        )

@router.get("/{kb_id}/files")
async def list_knowledge_base_files(kb_id: str):
    """获取知识库文件列表"""
    kb_path = DATA_DIR / kb_id
    if not kb_path.exists() or not kb_path.is_dir():
        raise HTTPException(status_code=404, detail="知识库不存在")
    
    files = []
    for file in kb_path.iterdir():
        if file.is_file():
            # 尝试读取元数据文件获取完整信息
            meta_file = META_DIR / kb_id / f"{file.name}.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as mf:
                    file_meta = json.load(mf)
                # 确保status字段存在且有效
                if 'status' not in file_meta:
                    file_meta['status'] = 'uploaded'
                files.append(file_meta)
            else:
                # 如果没有元数据文件，创建基本元数据
                files.append({
                    "name": file.name,
                    "size": file.stat().st_size,
                    "created_at": datetime.fromtimestamp(file.stat().st_ctime).isoformat(),
                    "updated_at": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    "status": "uploaded",
                    "path": str(file.relative_to(DATA_DIR.parent.parent)),
                    "kb_id": kb_id
                })
    return files

@router.post("/{kb_id}/process-batch")
async def process_batch_files(
    request: Request,
    kb_id: str,
    payload: Dict = Body(...)
):
    """批量处理知识库文件"""
    try:
        # 验证知识库存在
        kb_path = DATA_DIR / kb_id
        if not kb_path.exists():
            raise HTTPException(status_code=404, detail="知识库不存在")
        
        # 获取请求参数
        embedding_model = payload.get("embedding_model_id", "bge-large-zh-v1.5")
        incremental = payload.get("incremental", True)
        # 从元数据获取向量存储路径
        meta_config_path = META_DIR / kb_id / "config.json"
        if not meta_config_path.exists():
            raise HTTPException(
                status_code=404,
                detail="知识库配置文件不存在"
            )
        
        with open(meta_config_path, encoding="utf-8") as f:
            config = json.load(f)
            vector_path = config.get("vector_path")
            if not vector_path:
                raise HTTPException(
                    status_code=500,
                    detail="配置中缺少向量存储路径"
                )
        
        # 确保使用绝对路径
        vector_path = str(DATA_DIR.parent / vector_path)

        # 在调用 process_documents 前处理路径
        # 针对Windows中文路径问题
        # vector_path = vector_path.encode('utf-8').decode('gbk') 
        # 确保向量存储目录存在且可写
        vector_path = os.path.abspath(vector_path)
        os.makedirs(vector_path, exist_ok=True)
        # 第一次运行时强制关闭增量模式
        if not os.path.exists(os.path.join(vector_path, "default__vector_store.json")):
            incremental = False
        
        # 获取请求体并验证
        try:
            request_data = await request.json()
            selected_files = request_data.get("selected_files", [])
            if not isinstance(selected_files, list):
                raise ValueError("selected_files必须是列表")
            if not selected_files:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "请选择要处理的文件",
                        "code": "NO_FILES_SELECTED",
                        "selected_files": selected_files
                    }
                )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "无效的JSON请求体",
                    "code": "INVALID_JSON"
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": str(e),
                    "code": "INVALID_REQUEST"
                }
            )
        
        # 创建临时文件夹
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 复制选中文件到临时文件夹
            for filename in selected_files:
                src_file = kb_path / filename
                if not src_file.exists():
                    continue
                dest_file = temp_dir / filename
                shutil.copy2(src_file, dest_file)
            
            # 批量处理临时文件夹中的文件
            process_documents(
                file_path=str(temp_dir),
                vector_path=vector_path,
                embedding_model_id=embedding_model,
                incremental=incremental,
                kb_id=kb_id
            )
            
            # 更新文件元数据状态
            for filename in selected_files:
                meta_file = META_DIR / kb_id / f"{filename}.json"
                if meta_file.exists():
                    with open(meta_file, "r+", encoding="utf-8") as f:
                        meta = json.load(f)
                        meta["status"] = "processed"
                        meta["updated_at"] = datetime.now().isoformat()
                        f.seek(0)
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                        f.truncate()
            
            # 更新知识库元数据
            meta_config_path = META_DIR / kb_id / "config.json"
            if meta_config_path.exists():
                with open(meta_config_path, "r+", encoding="utf-8") as f:
                    config = json.load(f)
                    config["updated_at"] = datetime.now().isoformat()
                    config["last_processed"] = datetime.now().isoformat()
                    config["processed_files"] = len(selected_files)
                    f.seek(0)
                    json.dump(config, f, ensure_ascii=False, indent=2)
                    f.truncate()
            
            return {
                "status": "completed",
                "processed_count": len(selected_files),
                "embedding_model": embedding_model,
                "kb_id": kb_id
            }
            
        finally:
            # 清理临时文件夹
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "kb_id": kb_id,
            "selected_files": selected_files,
            "temp_dir": str(temp_dir) if 'temp_dir' in locals() else None
        }
        logging.error(f"批量处理失败: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "批量处理失败",
                "error": str(e),
                "kb_id": kb_id
            }
        )

@router.delete("/{kb_id}/files/{filename}")
async def delete_knowledge_base_file(kb_id: str, filename: str):
    """删除知识库中的文件"""
    try:
        # 解码文件名
        decoded_filename = urllib.parse.unquote(filename)
        
        # 验证知识库存在
        kb_path = DATA_DIR / kb_id
        if not kb_path.exists():
            raise HTTPException(status_code=404, detail="知识库不存在")
        
        # 验证文件存在
        file_path = kb_path / decoded_filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 删除文件
        file_path.unlink()
        
        # 删除元数据文件
        meta_file = META_DIR / kb_id / f"{decoded_filename}.json"
        if meta_file.exists():
            meta_file.unlink()
        
        return {
            "status": "deleted",
            "filename": decoded_filename,
            "kb_id": kb_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文件删除失败: {str(e)}"
        )

@router.post("/{kb_id}/files/{filename}/process")
async def process_file(
    kb_id: str,
    filename: str,
    request: Request
):
    """处理知识库文件生成向量"""
    try:
        # 解码文件名
        decoded_filename = urllib.parse.unquote(filename)
        
        # 验证知识库存在
        kb_path = DATA_DIR / kb_id
        if not kb_path.exists():
            raise HTTPException(status_code=404, detail="知识库不存在")
        
        # 验证文件存在
        file_path = kb_path / decoded_filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 获取请求参数
        payload = await request.json()
        embedding_model = payload.get("embedding_model_id", "bge-large-zh-v1.5")
        incremental = payload.get("incremental", True)
        
        # 从元数据获取向量存储路径
        meta_config_path = META_DIR / kb_id / "config.json"
        if not meta_config_path.exists():
            raise HTTPException(
                status_code=404,
                detail="知识库配置文件不存在"
            )
        
        with open(meta_config_path, encoding="utf-8") as f:
            config = json.load(f)
            vector_path = config.get("vector_path")
            if not vector_path:
                raise HTTPException(
                    status_code=500,
                    detail="配置中缺少向量存储路径"
                )
        
        # 确保使用绝对路径
        vector_path = str(DATA_DIR.parent / vector_path)
        # 调用文件处理服务
        try:
            process_single_file(
                kb_id=kb_id,
                org_file=str(file_path),
                vec_dir=vector_path,
                embedding_model_id=embedding_model,
                incremental=incremental,
                max_retries=3
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"文件处理失败: {str(e)}"
            )
            # 更新文件状态
        
        # 更新文件状态
        meta_file = META_DIR / kb_id / f"{decoded_filename}.json"
        if meta_file.exists():
            with open(meta_file, "r+", encoding="utf-8") as f:
                meta = json.load(f)
                meta["status"] = "processed"
                meta["updated_at"] = datetime.now().isoformat()
                f.seek(0)
                json.dump(meta, f, ensure_ascii=False, indent=2)
                f.truncate()
        
        return {
            "status": "processing",
            "filename": decoded_filename,
            "kb_id": kb_id,
            "embedding_model": embedding_model,
            "incremental": incremental
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的请求参数")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文件处理失败: {str(e)}"
        )
