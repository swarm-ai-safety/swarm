"""Trace ID middleware for per-request correlation."""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class TraceIDMiddleware(BaseHTTPMiddleware):
    """Attach a trace ID to every request.

    Reads ``X-Trace-ID`` from the incoming request headers.  If absent,
    generates a new UUID-4.  The trace ID is stored on ``request.state.trace_id``
    and echoed back in the response ``X-Trace-ID`` header.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())
        request.state.trace_id = trace_id

        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response
