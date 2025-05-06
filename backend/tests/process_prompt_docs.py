
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# 设置环境变量确保相对导入正常工作
os.environ["PYTHONPATH"] = project_root

from backend.utils import process_documents

def main():
    # 文档路径
    doc_path = "backend/data/prompt"
    # 向量存储路径
    vector_path = "backend/vectorstore/prompt"
    # 使用bge-large-zh-v1.5嵌入模型
    embedding_model_id = "bge-large-zh-v1.5"
    
    print(f"处理文档路径: {doc_path}")
    print(f"向量存储路径: {vector_path}")
    print(f"使用嵌入模型: {embedding_model_id}")
    
    try:
        print("开始处理文档...")
        # 确保向量存储目录存在
        from pathlib import Path
        Path(vector_path).mkdir(parents=True, exist_ok=True)
        print(f"确保向量存储目录存在: {vector_path}")

        # 处理文档
        index, results = process_documents(
            doc_path, 
            vector_path, 
            embedding_model_id,
            incremental=False  # 强制重新创建索引
        )
        
        print("文档处理完成")
        print("生成的向量文件:")
        for f in Path(vector_path).glob('*'):
            print(f"  - {f.name} (大小: {f.stat().st_size} bytes)")
        
        print("处理结果:")
        for result in results:
            print(f"{result['doc_id']}: {result['status']} - {result['message']}")
    except Exception as e:
        print(f"文档处理失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
