import sys
import json
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
import httpx
from datasets import Dataset, load_from_disk
from ragas import evaluate
from ragas.metrics import ContextRelevance, Faithfulness, AnswerRelevancy
from ragas.llms.base import BaseRagasLLM
from llama_index.llms.openai_like import OpenAILike

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("NvidiaDeepSeek")

# 项目根目录设置
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
try:
    from backend import utils
except ImportError:
    logger.warning("未找到backend.utils模块")

# ================== 配置加载 ==================
CONFIG_PATH = "config.json"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    MODEL_CONFIG = json.load(f)

# ================== 智能并发控制器 ==================
class AdaptiveConcurrencyController:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(1)  # 初始并发数
        self.current_delay = 0.0
        self.max_permits = 2  # 最大并发数

    async def adjust(self, success: bool):
        """动态调整并发策略"""
        if success:
            self.current_delay = max(0.0, self.current_delay - 0.5)
            if self.semaphore._value < self.max_permits:
                new_permits = self.semaphore._value + 1
                self.semaphore = asyncio.Semaphore(new_permits)
                logger.info(f"✅ 并发数提升至 {new_permits}")
        else:
            self.current_delay += 1.0
            new_permits = max(1, self.semaphore._value - 1)
            self.semaphore = asyncio.Semaphore(new_permits)
            logger.warning(f"⚠️ 并发数降至 {new_permits} 当前延迟 {self.current_delay:.1f}s")

    async def __aenter__(self):
        await asyncio.sleep(self.current_delay)
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, *args):
        self.semaphore.release()

# ================== NVIDIA DeepSeek适配器 ==================
class NvidiaDeepSeekRagasLLM(BaseRagasLLM):
    def __init__(self, deepseek_llm: OpenAILike):
        super().__init__()
        self.llm = deepseek_llm
        self.concurrency_ctl = AdaptiveConcurrencyController()
        self._validate_endpoint()
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm.api_key}",
            "x-api-source": "python-sdk",
            "accept": "application/json"
        }
        logger.info(f"🔧 初始化成功 | 模型: {self.llm.model}")

    def _validate_endpoint(self):
        """严格验证API端点"""
        from urllib.parse import urlparse
        
        parsed = urlparse(self.llm.api_base)
        if parsed.path not in ["", "/"]:
            logger.warning(f"⚠️ 配置URL包含冗余路径: {parsed.path}")
        
        # 生成规范化的API端点
        self.llm.api_base = f"{parsed.scheme}://{parsed.netloc}/v1/chat/completions"
        logger.info(f"✅ 最终API端点: {self.llm.api_base}")

        # 预检端点有效性
        try:
            response = httpx.get(f"{parsed.scheme}://{parsed.netloc}/v1/models", timeout=10)
            if response.status_code != 200:
                raise ValueError(f"API端点验证失败: {response.status_code}")
        except Exception as e:
            logger.critical(f"🔴 端点验证失败: {str(e)}")
            exit(1)

    def generate_text(self, prompt: str, **kwargs) -> str:
        """同步生成"""
        params = self._build_params(prompt, kwargs)
        return self._execute_request(params)

    async def agenerate_text(self, prompt: str, **kwargs) -> str:
        """异步生成"""
        params = self._build_params(prompt, kwargs)
        async with self.concurrency_ctl:
            try:
                result = await self._execute_async_request(params)
                await self.concurrency_ctl.adjust(True)
                return result
            except Exception as e:
                await self.concurrency_ctl.adjust(False)
                raise

    def _build_params(self, prompt: str, kwargs: dict) -> dict:
        """构建安全参数"""
        return {
            "model": "deepseek-ai/deepseek-r1",
            "messages": [{
                "role": "user",
                "content": self._clean_prompt(prompt)
            }],
            "temperature": 0.2,
            "max_tokens": 1024,
            "top_p": 0.9,
            "presence_penalty": 0.5,
            "stream": False
        }

    def _clean_prompt(self, prompt: str) -> str:
        """清理复杂prompt结构"""
        if isinstance(prompt, tuple):
            return prompt[1].split("Output:")[0].strip()
        return str(prompt).split("Output:")[0].strip()

    @retry(stop=stop_after_attempt(3),
           wait=wait_random_exponential(min=1, max=30),
           retry=retry_if_exception_type((httpx.HTTPError, json.JSONDecodeError)))
    def _execute_request(self, params: dict) -> str:
        """执行同步请求"""
        try:
            response = httpx.post(
                self.llm.api_base,
                headers=self.headers,
                json=params,
                timeout=60
            )
            return self._process_response(response, params)
        except httpx.HTTPError as e:
            self._log_error(e, params)
            raise

    @retry(stop=stop_after_attempt(3),
           wait=wait_random_exponential(min=1, max=30),
           retry=retry_if_exception_type((httpx.HTTPError, json.JSONDecodeError)))
    async def _execute_async_request(self, params: dict) -> str:
        """执行异步请求"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.llm.api_base,
                    headers=self.headers,
                    json=params,
                    timeout=60
                )
                return self._process_response(response, params)
            except httpx.HTTPError as e:
                self._log_error(e, params)
                raise

    def _process_response(self, response: httpx.Response, params: dict) -> str:
        """响应处理"""
        try:
            response.raise_for_status()
            data = response.json()
            return json.dumps(data["choices"][0]["message"]["content"])
        except Exception as e:
            logger.error(f"🚨 响应处理失败 | URL: {response.url}")
            logger.error(f"⚙️ 请求参数: {json.dumps(params, indent=2)}")
            logger.error(f"📄 响应内容: {response.text[:300]}...")
            raise

    def _log_error(self, e: httpx.HTTPError, params: dict):
        """错误日志记录"""
        logger.error(f"""
        === 请求失败详情 ===
        URL: {e.request.url}
        状态码: {e.response.status_code}
        请求头: {dict(e.request.headers)}
        参数: {json.dumps(params, indent=2)}
        响应内容: {e.response.text[:300]}...
        """)

# ================== 初始化配置 ==================
os.environ["OPENAI_API_KEY"] = "invalid"  # 确保禁用OpenAI

# 初始化模型
config = MODEL_CONFIG["deepseek"]
deepseek_llm = OpenAILike(
    api_base=config['url'],
    api_key=config['api_key'],
    model="deepseek-r1",
    is_chat_model=True,
    temperature=0.2,
    max_tokens=1024,
    timeout=60
)

# ================== 评估配置 ==================
ragas_llm = NvidiaDeepSeekRagasLLM(deepseek_llm)
metrics = [
    ContextRelevance(llm=ragas_llm),
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm)
]

# ================== 数据集函数 ==================
def prepare_eval_dataset():
    """准备评估数据集（首次运行需取消注释）"""
    knowledge_base_id = "信贷业务"
    embedding_model_id = "huggingface_bge-large-zh-v1.5"
    eval_questions = ["贷后管理包含哪些主要工作内容？"]

    answers, contexts = [], []
    for q in eval_questions:
        try:
            query_engine = utils.load_vector_index(
                knowledge_base_id, 
                embedding_model_id
            ).as_query_engine(llm=deepseek_llm)
            response = query_engine.query(q)
            answers.append(response.response.strip())
            contexts.append([node.text for node in response.source_nodes])
        except Exception as e:
            logger.error(f"生成答案失败: {str(e)}")
            answers.append("")
            contexts.append([])

    eval_dataset = Dataset.from_dict({
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts
    })
    eval_dataset.save_to_disk("eval_dataset")
    logger.info("💾 评估数据集已保存")

# ================== 评估执行 ==================
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def run_evaluate():
    """执行评估流程"""
    # API连通性测试
    try:
        test_prompt = "生成测试响应"
        test_res = ragas_llm.generate_text(test_prompt)
        logger.info(f"🟢 API测试响应: {test_res[:50]}...")
    except Exception as e:
        logger.critical(f"🔴 API测试失败: {str(e)}")
        exit(1)

    # 加载数据集
    try:
        eval_dataset = load_from_disk("eval_dataset")
        logger.info(f"📂 加载数据集成功 | 样本数: {len(eval_dataset)}")
    except Exception as e:
        logger.error(f"数据集加载失败: {str(e)}")
        exit(1)

    # 执行评估
    try:
        result = evaluate(
            eval_dataset,
            metrics=metrics,
            llm=ragas_llm,
            raise_exceptions=False,
            timeout=300
        )
    except Exception as e:
        logger.critical(f"评估流程异常终止: {str(e)}")
        exit(1)

    # 结果安全处理
    logger.info("\n" + " 评估报告 ".center(50, "="))
    
    score_map = {
        'context_relevance': 0.0,
        'faithfulness': 0.0,
        'answer_relevancy': 0.0
    }
    
    for key in score_map.keys():
        if key in result:
            score_map[key] = result[key].mean(skipna=True)
    
    logger.info(f"上下文相关性: {score_map['context_relevance']:.2%}")
    logger.info(f"回答忠实度: {score_map['faithfulness']:.2%}")
    logger.info(f"答案相关度: {score_map['answer_relevancy']:.2%}")

    logger.info("\n详细结果:")
    print(result.to_pandas().to_markdown(index=False))

if __name__ == "__main__":
    # prepare_eval_dataset()  # 首次运行取消注释
    run_evaluate()