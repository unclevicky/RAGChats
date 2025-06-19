from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import time
import logging
from fastapi import HTTPException, status
from typing import Dict, Set, Optional

class StreamBufferMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Transfer-Encoding"] = "chunked"
        return response

class ConcurrencyLimiterMiddleware(BaseHTTPMiddleware):
    """
    限制各类API的并发请求数，防止系统过载
    """
    def __init__(
        self, 
        app, 
        chat_limit: int = 5,
        kb_creation_limit: int = 3,
        file_process_limit: int = 2
    ):
        super().__init__(app)
        # 不同类型请求的并发限制
        self.limits = {
            "chat": chat_limit,
            "kb_creation": kb_creation_limit,
            "file_process": file_process_limit,
        }
        
        # 当前活跃的请求
        self.active_requests: Dict[str, Set[str]] = {
            "chat": set(),
            "kb_creation": set(),
            "file_process": set(),
        }
        
        # 请求计数和限流
        self.request_counts: Dict[str, int] = {
            "chat": 0,
            "kb_creation": 0,
            "file_process": 0
        }
        
        # 用于请求锁定的信号量
        self.locks = {
            key: asyncio.Semaphore(value) 
            for key, value in self.limits.items()
        }
        
        # 最后清理时间
        self.last_cleanup = time.time()
        
        # 用于调试
        logging.info(f"初始化并发限制中间件: {self.limits}")
    
    def _get_request_type(self, request: Request) -> Optional[str]:
        """根据请求路径判断请求类型"""
        path = request.url.path
        method = request.method
        
        if path.startswith("/api/chat") and method == "POST":
            return "chat"
        elif "knowledge-bases" in path and "create" in path:
            return "kb_creation"
        elif "process" in path:
            return "file_process"
        return None
    
    async def dispatch(self, request: Request, call_next):
        # 获取请求类型
        request_type = self._get_request_type(request)
        request_id = f"{request.client.host}:{id(request)}"
        
        # 如果是受限制的请求类型，则进行限流
        if request_type:
            try:
                # 尝试获取信号量
                async with self.locks[request_type]:
                    # 记录活跃请求
                    self.active_requests[request_type].add(request_id)
                    self.request_counts[request_type] += 1
                    
                    # 记录请求开始
                    start_time = time.time()
                    logging.info(f"开始处理 {request_type} 请求: {request.url.path} [当前并发: {len(self.active_requests[request_type])}]")
                    
                    # 处理请求
                    response = await call_next(request)
                    
                    # 记录请求结束
                    duration = time.time() - start_time
                    logging.info(f"完成处理 {request_type} 请求: {request.url.path} [耗时: {duration:.2f}s]")
                    
                    # 清理活跃请求
                    self.active_requests[request_type].remove(request_id)
                    
                    return response
            except asyncio.TimeoutError:
                logging.error(f"请求超时: {request.url.path}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="服务暂时不可用，请稍后重试"
                )
        else:
            # 不需要限流的请求直接处理
            return await call_next(request)

class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    跟踪请求处理时间，记录慢请求
    """
    def __init__(self, app, slow_request_threshold: float = 5.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.logger = logging.getLogger("request_tracker")
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 添加请求ID便于追踪
        request_id = f"{time.time()}-{id(request)}"
        request.state.request_id = request_id
        
        # 请求预处理日志
        self.logger.info(f"[{request_id}] 开始处理请求: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # 请求后处理
            duration = time.time() - start_time
            
            # 记录慢请求
            if duration > self.slow_request_threshold:
                self.logger.warning(
                    f"[{request_id}] 慢请求: {request.method} {request.url.path} - {duration:.2f}s"
                )
            else:
                self.logger.info(
                    f"[{request_id}] 完成请求: {request.method} {request.url.path} - {duration:.2f}s"
                )
                
            # 添加服务器处理时间响应头
            response.headers["X-Process-Time"] = str(duration)
            response.headers["X-Request-ID"] = request_id
            
            return response
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"[{request_id}] 请求异常: {request.method} {request.url.path} - {str(e)} - {duration:.2f}s"
            )
            raise