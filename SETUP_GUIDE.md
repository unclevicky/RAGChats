# RAGChats 安装与使用指南

## 环境要求

- Python 3.12.11
- Node.js 16+ (前端开发)
- 足够的磁盘空间用于存储向量数据库和模型缓存

## 安装步骤

### 1. 克隆项目

```bash
git clone <项目仓库URL>
cd RAGChats
```

### 2. 后端环境设置

推荐使用虚拟环境:

```bash
# 创建虚拟环境
python -m venv ragchat_env
# Windows激活虚拟环境
ragchat_env\Scripts\activate
# Linux/Mac激活虚拟环境
# source ragchat_env/bin/activate

# 安装后端依赖
cd backend
pip install -r requirements.txt
cd ..
```

### 3. 前端环境设置

```bash
cd frontend
npm install
cd ..
```

## 启动项目

### 方法一：使用脚本启动（推荐）

直接运行提供的启动脚本:

```bash
start_ragchat.bat  # Windows
```

该脚本会自动:
- 检查Python环境
- 安装必要的依赖
- 创建必要的目录
- 启动后端服务 (http://localhost:8000)
- 启动前端服务 (http://localhost:5173)

### 方法二：手动启动

1. 启动后端服务:

```bash
cd backend
set PYTHONPATH=..  # Windows
# export PYTHONPATH=..  # Linux/Mac
uvicorn main:app --reload --port 8000
```

2. 在另一个终端启动前端服务:

```bash
cd frontend
npm run dev
```

## 可能遇到的问题及解决方案

### 1. ModuleNotFoundError: No module named 'llama_index.vector_stores'

这是由于llama-index库结构变化导致的。解决方案:

```bash
pip install llama-index==0.12.42 llama-index-vector-stores-faiss==0.4.0
```

### 2. 找不到zh_core_web_sm模型

运行以下命令安装:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.7.0/zh_core_web_sm-3.7.0-py3-none-any.whl
```

### 3. 前端依赖安装失败

尝试使用:

```bash
cd frontend
npm install --legacy-peer-deps
```

## 项目结构

```
RAGChats/
├── backend/             # 后端代码
│   ├── routes/          # API路由
│   ├── utils.py         # 工具函数
│   ├── main.py          # 入口文件
│   └── requirements.txt # 后端依赖
├── frontend/            # 前端代码
│   ├── src/             # 源代码
│   └── package.json     # 前端依赖
└── start_ragchat.bat    # 启动脚本
```

## 注意事项

- 确保端口8000和5173未被其他应用占用
- 首次运行时，系统会自动创建必要的目录结构
- 使用大型模型或处理大量文档时，请确保有足够的系统资源 