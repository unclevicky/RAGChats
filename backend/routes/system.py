
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict
from pydantic import BaseModel
import json
import os
from pathlib import Path
import importlib
import sys

router = APIRouter()

def get_config_path(module_name):
    # 判断是否为打包环境
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS  # 单文件夹模式下指向 dist/程序名 目录
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, f"{module_name}.json")

"""这种写法不兼容打包后exe运行模式
# 助手配置文件路径
ASSISTANTS_PATH = Path(__file__).parent.parent / "assistants.py"

# 配置文件路径
CONFIG_PATH = Path(__file__).parent.parent / "config.py"
"""
""" 换一种写法
ASSISTANTS_PATH = Path(project_root) / "assistants.py"
CONFIG_PATH = Path(project_root)/ "config.py"
"""
ASSISTANTS_PATH = Path(get_config_path("assistants"))
CONFIG_PATH = Path(get_config_path("config"))

def load_config():
    """从config.py加载配置"""
    """ 修改为动态加载
    from backend.config import MODEL_CONFIG
    return MODEL_CONFIG.copy()
    """
    """改为json
    config_module = importlib.import_module('backend.config')
    importlib.reload(config_module)  # 强制重新加载模块
    return config_module.MODEL_CONFIG.copy()
    """
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置失败: {str(e)}")
        return {}


def save_config(config):
    """保存配置到config.py"""
    """改为json
    try:
        print(f"尝试保存配置到: {CONFIG_PATH}")  # 调试日志
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            content = f"MODEL_CONFIG = {json.dumps(config, indent=4, ensure_ascii=False)}\n"
            f.write(content)
            print(f"配置保存成功，内容长度: {len(content)}")  # 调试日志
        return True
    except Exception as e:
        print(f"保存配置失败: {str(e)}")  # 调试日志
        return False
    """
    try:
        print(f"尝试保存配置到: {CONFIG_PATH}")  # 调试日志
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"配置保存成功")  # 调试日志
        return True
    except Exception as e:
        print(f"保存配置失败: {str(e)}")  # 调试日志
        return False

class Model(BaseModel):
    name: str
    type: str
    url: str
    model: str
    api_key: str
    provider: str

class Assistant(BaseModel):
    name: str
    description: str = ""
    model: str
    knowledge_base: str
    system_prompt: str
    embedding: str = "text-embedding-3-small"

# 临时模拟数据存储
models_db = []

def load_assistants():
    """加载助手配置"""
    if not ASSISTANTS_PATH.exists():
        return {}
    """修改为动态加载
    from backend.assistants import ASSISTANTS
    return ASSISTANTS.copy()
    """
    '''改为json
    # 换一种写法
    assistants_module = importlib.import_module('backend.assistants')
    importlib.reload(assistants_module)  # 强制重新加载模块
    return assistants_module.ASSISTANTS.copy()
    '''
    try:
        with open(ASSISTANTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置失败: {str(e)}")
        return {}

def save_assistants(assistants):
    """保存助手配置"""
    '''改为json
    content = f"ASSISTANTS = {json.dumps(assistants, indent=4, ensure_ascii=False)}"
    ASSISTANTS_PATH.write_text(content, encoding='utf-8')
    '''
    with open(ASSISTANTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(assistants, f, indent=4, ensure_ascii=False)

@router.get("/assistants", response_model=List[Assistant])
async def get_assistants():
    """获取所有助手配置"""
    try:
        assistants = load_assistants()
        result = []
        for name, cfg in assistants.items():
            result.append(Assistant(
                name=name,
                description=cfg.get("description", ""),
                model=cfg.get("model", ""),
                knowledge_base=cfg.get("knowledge_base", ""),
                system_prompt=cfg.get("system_prompt", ""),
                embedding=cfg.get("embedding", "text-embedding-3-small")
            ))
        return result
    except Exception as e:
        print(f"获取助手列表失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取助手列表失败: {str(e)}"
        )

@router.put("/assistants/{name}")
async def update_assistant(name: str, assistant: Assistant):
    """更新或创建助手配置"""
    if not assistant.name or not assistant.name.strip():
        raise HTTPException(status_code=400, detail="助手名称不能为空")
    
    assistants = load_assistants()
    assistants[name] = {
        "description": assistant.description,
        "model": assistant.model,
        "knowledge_base": assistant.knowledge_base,
        "system_prompt": assistant.system_prompt,
        "embedding": assistant.embedding
    }
    save_assistants(assistants)
    
    '''改为json
    # 强制重新加载模块
    import importlib
    from backend import assistants
    importlib.reload(assistants)
    '''
    
    # 文件同步
    try:
        if os.name == 'nt':  # Windows
            import time
            time.sleep(0.5)
            if os.path.exists(ASSISTANTS_PATH):
                os.stat(ASSISTANTS_PATH)
        else:  # Unix-like
            os.sync()
    except Exception as e:
        print(f"文件同步失败: {str(e)}")
    
    # 返回更新后的完整助手列表
    updated_assistants = load_assistants()
    result = []
    for name, cfg in updated_assistants.items():
            result.append(Assistant(
                name=name,
                description=cfg.get("description", ""),
                model=cfg.get("model", ""),
                knowledge_base=cfg.get("knowledge_base", ""),
                system_prompt=cfg.get("system_prompt", ""),
                embedding=cfg.get("embedding", "text-embedding-3-small")
            ))
    
    return {
        "message": "Assistant updated",
        "assistants": result
    }

@router.delete("/assistants/{name}")
async def delete_assistant(name: str):
    """删除助手配置"""
    try:
        assistants = load_assistants()
        if name not in assistants:
            raise HTTPException(status_code=404, detail="Assistant not found")
        
        # 备份要删除的助手
        deleted_assistant = {
            "name": name,
            "config": assistants[name]
        }
        
        del assistants[name]
    
        # 重试机制
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            # 保存配置
            save_assistants(assistants)
            
            '''改为json
            # 强制重新加载模块
            import importlib
            from backend import assistants
            importlib.reload(assistants)
            '''
            # 文件同步
            try:
                if os.name == 'nt':  # Windows
                    import time
                    time.sleep(retry_delay)
                    if os.path.exists(ASSISTANTS_PATH):
                        os.stat(ASSISTANTS_PATH)
                else:  # Unix-like
                    os.sync()
            except Exception as e:
                print(f"文件同步失败: {str(e)}")
            
            # 验证删除结果
            updated_assistants = load_assistants()
            if name not in updated_assistants:
                # 返回剩余助手列表
                remaining_assistants = []
                for name, cfg in updated_assistants.items():
                    remaining_assistants.append(Assistant(
                        name=name,
                        description=cfg.get("description", ""),
                        model=cfg.get("model", ""),
                        knowledge_base=cfg.get("knowledge_base", ""),
                        system_prompt=cfg.get("system_prompt", "")
                    ))
                
                return {
                    "message": "Assistant deleted",
                    "deleted": deleted_assistant,
                    "remaining": remaining_assistants
                }
            
            if attempt < max_retries - 1:
                print(f"删除验证失败，第{attempt+1}次重试，等待{retry_delay}秒...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 1.5  # 指数退避
        
        # 所有重试都失败
        raise HTTPException(
            status_code=500,
            detail={
                "message": "删除操作未完全生效，可能是服务未完全重启",
                "error": "配置同步延迟",
                "suggestion": "请等待几秒后重试或手动重启服务"
            }
        )
            
    except Exception as e:
        print(f"删除助手{name}失败: {str(e)}")
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
        raise

@router.get("/models", response_model=List[Model])
async def get_models():
    """获取所有模型配置"""
    config = load_config()
    result = []
    for name, cfg in config.items():
        result.append(Model(
            name=name,
            type=cfg.get("type", ""),
            url=cfg.get("url", ""),
            model=cfg.get("model", ""),
            api_key=cfg.get("api_key", ""),
            provider=cfg.get("provider", "")
        ))
    return result

@router.put("/models/{name}")
async def update_model(name: str, model: Model):
    """更新或创建模型配置"""
    config = load_config()
    
    # 验证模型名称
    if not model.name or not model.name.strip():
        raise HTTPException(
            status_code=400,
            detail="模型名称不能为空"
        )
    
    # 转换模型数据为配置格式
    model_config = {
        "url": model.url,
        "model": model.model,
        "api_key": model.api_key,
        "type": model.type,
        "provider": model.provider
    }
    
    config[name] = model_config
    if not save_config(config):
        raise HTTPException(status_code=500, detail="保存配置失败")
    
    '''改为json
    # 强制重新加载模块
    import importlib
    from backend import config
    importlib.reload(config)
    '''
    # 更可靠的文件同步方式
    try:
        if os.name == 'nt':  # Windows
            import time
            time.sleep(1.0)
            if os.path.exists(CONFIG_PATH):
                os.stat(CONFIG_PATH)
        else:
            os.sync()
        print("文件同步成功")
    except Exception as e:
        print(f"文件同步失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"文件同步失败: {str(e)}"
        )
    
    # 双重验证
    file_config = load_config()
    # print(f"验证配置 - 文件: {file_config.keys()}, 内存: {config.items()}")
    
    # 返回最新数据
    result = []
    for name, cfg in config.items():
        result.append(Model(
            name=name,
            type=cfg.get("type", ""),
            url=cfg.get("url", ""),
            model=cfg.get("model", ""),
            api_key=cfg.get("api_key", ""),
            provider=cfg.get("provider", "")
        ))
    
    return {
        "message": "Model updated",
        "models": result,
        "verified": name in config
    }

@router.delete("/models/{name}")
async def delete_model(name: str):
    """删除模型配置"""
    try:
        config = load_config()
        if name not in config:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # 备份要删除的配置
        deleted_config = config[name]
        del config[name]
        
        # 验证保存结果，改进的重试机制
        max_retries = 3
        retry_delay = 1.0  # 初始重试延迟1秒
        
        for attempt in range(max_retries):
            if not save_config(config):
                raise HTTPException(status_code=500, detail="保存配置失败")
            
            '''改为json
            # 强制重新加载模块
            import importlib
            from backend import config
            importlib.reload(config)
            '''
            # 重新加载验证
            new_config = load_config()
            print(f"验证配置: {new_config.keys()}")
            
            # 更可靠的文件同步方式
            try:
                if os.name == 'nt':  # Windows
                    # 确保文件已关闭并同步
                    import time
                    time.sleep(1.0)  # 增加等待时间
                    if os.path.exists(CONFIG_PATH):
                        os.stat(CONFIG_PATH)  # 强制刷新文件状态
                else:  # Unix-like
                    os.sync()
                print("文件同步成功")
            except Exception as e:
                print(f"文件同步失败: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"文件同步失败: {str(e)}"
                )
            
            # 双重验证
            if name not in new_config and name not in config:
                return {
                    "message": "Model deleted",
                    "deleted": deleted_config,
                    "remaining": list(new_config.keys())
                }
            else:
                print(f"配置不一致 - 文件内容: {new_config.keys()}, 内存内容: {config.keys()}")
            
            if attempt < max_retries - 1:
                print(f"删除验证失败，第{attempt+1}次重试，等待{retry_delay}秒...")
                import time
                time.sleep(retry_delay)
                retry_delay *= 1.5  # 指数退避
        
        # 所有重试都失败
        from backend import config
        print(f"最终删除验证失败，模型仍存在: {name}")
        print(f"文件配置: {new_config.keys()}")
        print(f"内存配置: {config.MODEL_CONFIG.keys()}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "删除操作未生效，可能是服务未完全重启",
                "remaining": list(new_config.keys()),
                "error": "配置同步延迟",
                "suggestion": "请等待几秒后重试或手动重启服务"
            }
        )
            
    except Exception as e:
        print(f"删除模型{name}失败: {str(e)}")
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
        raise
