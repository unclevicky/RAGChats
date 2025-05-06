
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_vector_index, create_rag_engine
from config import MODEL_CONFIG

async def test_chat_flow():
    # 测试数据
    knowledge_base_id = "prompt"  # 使用项目中已有的知识库
    chat_model_id = "deepseek"  # 使用配置中已有的模型
    embedding_model_id = "bge-large-zh-v1.5"  # 嵌入模型ID
    
    try:
        print("1. 加载向量索引...")
        print(f"知识库ID: {knowledge_base_id}")
        vector_index = load_vector_index(knowledge_base_id)
        print(f"向量索引类型: {type(vector_index)}")
        print("向量索引加载成功")
        
        print("\n2. 创建RAG引擎...")
        print(f"模型ID: {chat_model_id}")
        print(f"模型配置: {MODEL_CONFIG.get(chat_model_id)}")
        rag_engine = create_rag_engine(vector_index, chat_model_id)
        print(f"RAG引擎类型: {type(rag_engine)}")
        print("RAG引擎创建成功")
        
        print("\n3. 测试查询...")
        test_question = "如何写好prompt?"
        print(f"测试问题: {test_question}")
        response = rag_engine.query(test_question)
        
        print("\n4. 测试结果:")
        print(f"响应类型: {type(response)}")
        print("响应内容:")
        print(response.response if hasattr(response, 'response') else response)
        
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_chat_flow())
