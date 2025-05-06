
# RAG本地知识问答系统

## 项目介绍
基于检索增强生成(RAG)的本地知识问答系统，支持：
- 知识库管理（上传/处理/删除文档）
- 多格式文档解析（PDF/Word/TXT等）
- 语义检索与问答
- 支持对话历史管理
- 支持主流大模型
- 支持自定义聊天助手

## 系统架构
```
前端(Vue.js) → 后端(FastAPI) → 向量数据库 → 大语言模型
```

## 项目目录结构
```
├── backend/                  # 后端服务
│   ├── logs/                 # 日志文件
│   ├── model_cache/          # 模型缓存
│   ├── routes/               # API路由
│   │   ├── chat.py           # 聊天相关接口
│   │   ├── knowledge.py      # 知识库管理接口
│   │   ├── system.py         # 系统配置接口
│   ├── tests/                # 单元测试
│   ├── __init__.py          # 包初始化
│   ├── assistants.py        # 助手管理
│   ├── config.py            # 配置管理
│   ├── main.py              # 服务入口
│   ├── requirements.txt     # Python依赖
│   └── utils.py            # 工具函数

├── frontend/                # 前端应用
│   ├── public/              # 静态资源
│   ├── src/                 # 源码目录
│   │   ├── api/             # API封装
│   │   ├── assets/          # 静态资源
│   │   ├── router/         # 路由配置
│   │   ├── store/          # 状态管理
│   │   ├── utils/          # 工具函数
│   │   ├── views/          # 页面组件
│   │   ├── App.vue         # 根组件
│   │   └── main.js         # 应用入口
│   ├── package.json        # 前端依赖
│   └── vite.config.js      # 构建配置

├── .gitignore              # Git忽略配置
├── Dockerfile              # 容器化配置
├── README.md               # 项目文档
├── run_tests.bat          # 测试脚本
└── start_all.bat          # 启动脚本
```

## 核心功能
### 前端功能
- 知识库管理界面
- 文件上传与处理状态跟踪
- 交互式问答界面
- 对话历史记录

### 后端功能
- 文档解析与分块
- 文本向量化处理
- 语义检索实现
- 大模型问答接口

## 安装指南

### 环境要求
- Python 3.8+
- Node.js 14+
- Redis (向量数据库)
- CPU/GPU(用于Embedding计算)

### 后端部署
```bash
cd backend
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件配置API密钥等

# 启动服务
python main.py
```

### 前端部署
```bash
cd frontend
npm install

# 开发模式
npm run dev

# 生产构建
npm run build
```

## 配置说明
### 关键配置项
- `EMBEDDING_MODEL`: 使用的嵌入模型
- `LLM_API_KEY`: 大模型API密钥
- `REDIS_URL`: 向量数据库地址
- `KNOWLEDGE_BASE_DIR`: 知识库存储路径

## 使用流程
1. 创建知识库
2. 上传文档文件
3. 等待文档处理完成
4. 开始问答交互

## 开发者指南
- 后端API文档：`/backend/docs`
- 前端组件文档：`/frontend/docs`
- 测试脚本：`/backend/tests`

## 常见问题
Q: 文档处理失败怎么办？
A: 检查日志文件`/backend/logs/backend.log`

Q: 问答响应慢？
A: 1. 检查CPU利用率 2. 优化分块大小
