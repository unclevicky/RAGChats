import sys
import json
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
import httpx
import numpy as np
from datasets import Dataset, load_from_disk
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    answer_correctness
)
from ragas.llms.base import BaseRagasLLM
from llama_index.llms.openai_like import OpenAILike
from langchain.schema import LLMResult, Generation

project_root = Path(__file__).parent

# 配置日志记录
def configure_logging():
    logger = logging.getLogger("NvidiaDeepSeek")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler("logs/rag_eval.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger

logger = configure_logging()
logger.info("✅ 日志系统初始化完成")

# ================== 配置加载 ==================
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    MODEL_CONFIG = json.load(f)

# ================== 智能并发控制器 ==================
class AdaptiveConcurrencyController:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(3)
        self.current_delay = 0.0
        self.max_permits = 5
        self.min_permits = 1

    async def adjust(self, success: bool):
        if success:
            self.current_delay = max(0.0, self.current_delay - 1.0)
            if self.semaphore._value < self.max_permits:
                new_permits = min(self.max_permits, self.semaphore._value + 1)
                self.semaphore = asyncio.Semaphore(new_permits)
                logger.info(f"✅ 并发数提升至 {new_permits}")
        else:
            self.current_delay += 2.0
            new_permits = max(self.min_permits, self.semaphore._value - 1)
            self.semaphore = asyncio.Semaphore(new_permits)
            logger.warning(f"⚠️ 并发数降至 {new_permits} | 延迟 {self.current_delay:.1f}s")

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
        from urllib.parse import urlparse
        parsed = urlparse(self.llm.api_base)
        self.llm.api_base = f"{parsed.scheme}://{parsed.netloc}/v1/chat/completions"
        logger.info(f"✅ API端点: {self.llm.api_base}")

     # 新增同步生成方法
    def generate_text(
        self,
        prompts: List[str],  # 必须保持为第一个位置参数
        n: int = 1,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs
        ) -> List[str]:
            """同步生成（参数顺序严格匹配）"""
            responses = []
            try:
                for prompt in prompts:
                    result = self._execute_sync(
                        prompt=prompt,
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop
                    )
                    responses.extend(result[:n])
            except Exception as e:
                logger.error(f"同步生成失败: {str(e)}")
                raise
            return responses

    async def agenerate_text(
        self,
        prompts: List[str],  # 必须保持为第一个位置参数
        n: int = 1,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        **kwargs  # 添加kwargs吸收额外参数
    ) -> List[str]:
        """异步生成（参数顺序严格匹配）"""
        responses = []
        async with self.concurrency_ctl:
            try:
                tasks = [
                    self._execute_async(
                        prompt=p,
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop
                    ) for p in prompts
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for res in results:
                    if isinstance(res, Exception):
                        responses.append("")
                        logger.error(f"异步生成失败: {str(res)}")
                    else:
                        responses.extend(res[:n])
                
                await self.concurrency_ctl.adjust(True)
            except Exception as e:
                await self.concurrency_ctl.adjust(False)
                logger.error(f"生成过程中发生未捕获异常: {str(e)}")
                raise
        return responses[:len(prompts)]  # 保持输出长度与输入一致

    def _build_params(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> dict:
        params = {
            "model": self.llm.model,
            "messages": [{
                "role": "user",
                "content": self._clean_prompt(prompt)
            }],
            "temperature": max(0.1, min(temperature, 1.0)),
            "max_tokens": min(max_tokens, 4096),
            "n": max(1, n),
            "stop_sequences": stop or []
        }
        params.update({
            "top_p": 0.9,
            "presence_penalty": 0.5
        })
        return params

    def _clean_prompt(self, prompt: str) -> str:
        if isinstance(prompt, (tuple, list)):
            return str(prompt[-1]).split("Output:")[0].strip()
        return str(prompt).split("Output:")[0].strip()

    @retry(stop=stop_after_attempt(3),
           wait=wait_random_exponential(min=1, max=30),
           retry=retry_if_exception_type((httpx.HTTPError, json.JSONDecodeError)))
    def _execute_sync(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> List[str]:
        try:
            params = self._build_params(prompt, n, temperature, max_tokens, stop)
            with httpx.Client(timeout=self.llm.timeout) as client:
                response = client.post(
                    self.llm.api_base,
                    headers=self.headers,
                    json=params
                )
                return self._process_response(response, params)
        except httpx.HTTPError as e:
            self._log_error(e, params)
            raise

    @retry(stop=stop_after_attempt(3),
       wait=wait_random_exponential(min=1, max=30),
       retry=retry_if_exception_type((httpx.HTTPError, json.JSONDecodeError, asyncio.TimeoutError)))
    async def _execute_async(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> List[str]:
        try:
            params = self._build_params(prompt, n, temperature, max_tokens, stop)
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=10.0),
                limits=httpx.Limits(max_connections=100)
            ) as client:
                response = await client.post(
                    self.llm.api_base,
                    headers=self.headers,
                    json=params
                )
                return self._process_response(response, params)
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            self._log_error(e, params)
            raise

    def _process_response(self, response: httpx.Response, params: dict) -> List[str]:
        try:
            response.raise_for_status()
            data = response.json()
            return [
                choice["message"]["content"].strip()
                for choice in data.get("choices", [])[:params.get("n", 1)]
            ]
        except Exception as e:
            logger.error(f"响应解析失败: {str(e)}")
            return []

    def _log_error(self, e: httpx.HTTPError, params: dict):
        error_info = {
            "url": str(e.request.url),
            "method": e.request.method,
            "status_code": e.response.status_code if e.response else None,
            "params": params
        }
        logger.error(f"API请求失败: {json.dumps(error_info, indent=2)}")

# ================== 初始化配置 ==================
os.environ["OPENAI_API_KEY"] = "invalid"

config = MODEL_CONFIG["deepseek"]
deepseek_llm = OpenAILike(
    api_base=config['url'],
    api_key=config['api_key'],
    model=config['model'],
    is_chat_model=True,
    temperature=0.3,
    max_tokens=1024,
    timeout=300,
    http_client=httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        limits=httpx.Limits(max_connections=100)
    )
)

# ================== 评估配置 ==================
ragas_llm = NvidiaDeepSeekRagasLLM(deepseek_llm)
metrics = [
    answer_relevancy,
    faithfulness,
    #answer_correctness
]

# ================== 数据集函数 ==================
def prepare_eval_dataset():
    try:
        sys.path.insert(0, str(project_root.parent))
        from backend import utils
        
        knowledge_base_id = "信贷业务"
        embedding_model_id = "huggingface_bge-large-zh-v1.5"
        eval_questions = [
            "信贷审批的特殊情形有哪些？",
            "如何计算贷款利息？",
            "逾期还款会有什么后果？"
        ]

        dataset_dict = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truths": []
        }

        for q in eval_questions:
            try:
                query_engine = utils.load_vector_index(
                    knowledge_base_id, 
                    embedding_model_id
                ).as_query_engine(llm=deepseek_llm)
                response = query_engine.query(q)
                
                dataset_dict["question"].append(q)
                dataset_dict["answer"].append(response.response.strip())
                dataset_dict["contexts"].append([node.text for node in response.source_nodes])
                dataset_dict["ground_truths"].append([])
                
            except Exception as e:
                logger.error(f"问题处理失败 [{q}]: {str(e)}")
                continue

        col_lengths = {k: len(v) for k, v in dataset_dict.items()}
        if len(set(col_lengths.values())) != 1:
            raise RuntimeError(f"列长度不一致: {col_lengths}")

        eval_dataset = Dataset.from_dict(dataset_dict)
        eval_dataset.save_to_disk(project_root / "eval_dataset")
        logger.info(f"💾 数据集已保存 | 样本数: {len(eval_dataset)}")

    except ImportError:
        logger.error("backend模块不可用")
        exit(1)

# ================== 评估执行 ==================
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def run_evaluate():
    """兼容新版Ragas的结果处理"""
    logger.info("启动评估流程".center(50, "="))
    
    try:
        eval_dataset = load_from_disk(project_root / "eval_dataset")
        logger.info(f"📂 加载数据集成功 | 样本数: {len(eval_dataset)}")
        
        required_columns = {'question', 'answer', 'contexts', 'ground_truths'}
        if not required_columns.issubset(eval_dataset.column_names):
            missing = required_columns - set(eval_dataset.column_names)
            raise ValueError(f"数据集结构损坏，缺失字段: {missing}")

    except Exception as e:
        logger.error(f"数据集错误: {str(e)}")
        exit(1)

    try:
        result = evaluate(
            eval_dataset,
            metrics=metrics,
            llm=ragas_llm,
            raise_exceptions=False
        )
        
        # 新版Ragas结果处理方式
        logger.info("\n" + " 评估报告 ".center(50, "="))
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            for metric in metrics:
                col_name = metric.name
                if col_name in df.columns:
                    scores = df[col_name].dropna()
                    if not scores.empty:
                        logger.info(f"{col_name}:")
                        logger.info(f"  平均值: {scores.mean():.2%}")
                        logger.info(f"  中位数: {scores.median():.2%}")
                        logger.info(f"  标准差: {scores.std():.2f}")
                    else:
                        logger.warning(f"{col_name}: 无有效数据")
        else:
            logger.error("无法识别评估结果格式")
            
    except Exception as e:
        logger.critical(f"评估失败: {str(e)}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    # prepare_eval_dataset()  # 首次运行时取消注释
    run_evaluate()