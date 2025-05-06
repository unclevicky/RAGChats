from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class StreamBufferMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Transfer-Encoding"] = "chunked"
        return response