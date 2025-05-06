
import os
import pytest
from fastapi.testclient import TestClient
import os
os.environ["APP_ENV"] = "test"
from backend.main import app
from backend.config_test import MODEL_CONFIG as TEST_CONFIG
app.state.MODEL_CONFIG = TEST_CONFIG
import shutil

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_test_environment():
    # 创建测试目录结构（使用绝对路径）
    os.makedirs("backend/data/test_kb_org", exist_ok=True)
    os.makedirs("backend/vectorstore/test_kb_vec", exist_ok=True)
    
    # 创建测试文件
    with open("backend/data/test_kb_org/test1.txt", "w") as f:
        f.write("test content")
    with open("backend/data/test_kb_org/test2.txt", "w") as f:
        f.write("test content")
    with open("backend/vectorstore/test_kb_vec/test1.txt", "w") as f:
        f.write("vector content")
    
    yield
    
    # 清理测试环境（使用绝对路径）
    shutil.rmtree("backend/data/test_kb_org", ignore_errors=True)
    shutil.rmtree("backend/vectorstore/test_kb_vec", ignore_errors=True)

def test_get_knowledge_bases(setup_test_environment):
    response = client.get("/knowledge-bases")
    assert response.status_code == 200
    kb_list = response.json()
    assert len(kb_list) > 0
    assert any(kb["id"] == "test_kb" for kb in kb_list)
    assert all("file_count" in kb for kb in kb_list)

def test_get_kb_files(setup_test_environment):
    response = client.get("/knowledge-bases/test_kb/files")
    assert response.status_code == 200
    files = response.json()
    assert len(files) == 2
    assert any(f["name"] == "test1.txt" and f["processed"] for f in files)
    assert any(f["name"] == "test2.txt" and not f["processed"] for f in files)
    assert all("size" in f and "created_at" in f for f in files)

def test_nonexistent_kb():
    response = client.get("/knowledge-bases/nonexistent/files")
    assert response.status_code == 200
    assert response.json() == []

def test_create_knowledge_base():
    # 测试新建知识库
    test_kb_name = "test_new_kb"
    response = client.post(
        "/knowledge-bases",
        data={"kb_name": test_kb_name}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert "kb_id" in result
    assert "org_dir" in result
    assert "vec_dir" in result
    
    # 验证目录是否创建
    assert os.path.exists(result["org_dir"])
    assert os.path.exists(result["vec_dir"])
    
    # 清理测试环境
    shutil.rmtree(result["org_dir"], ignore_errors=True)
    shutil.rmtree(result["vec_dir"], ignore_errors=True)

def test_process_single_file(setup_test_environment):
    # 测试单个文件解析接口
    test_file = "test2.txt"
    response = client.post(
        f"/knowledge-bases/test_kb/files/{test_file}/process"
    )
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    
    # 验证文件状态更新
    files_response = client.get("/knowledge-bases/test_kb/files")
    assert files_response.status_code == 200
    files = files_response.json()
    processed_file = next(f for f in files if f["name"] == test_file)
    assert processed_file["processed"] is True
    
    # 验证向量文件是否创建
    assert os.path.exists("backend/vectorstore/test_kb_vec/test2.txt")