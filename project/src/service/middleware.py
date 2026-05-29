import time
import logging
import uuid
from typing import Dict, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

metrics: Dict[str, any] = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "response_times_seconds": [],
    "start_time": time.time()
}

class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        metrics["requests_total"] += 1
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            metrics["response_times_seconds"].append(process_time)
            if 200 <= response.status_code < 400:
                metrics["requests_success"] += 1
            else:
                metrics["requests_error"] += 1
                
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} | "
                f"Status: {response.status_code} | Time: {process_time:.4f}s"
            )
            
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            metrics["requests_error"] += 1
            logger.error(f"[{request_id}] Unhandled exception: {str(e)}", exc_info=True)
            raise