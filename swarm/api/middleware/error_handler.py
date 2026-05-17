"""Structured error exception handlers."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

_STATUS_NAMES: dict[int, str] = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
}


def _get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "unknown")


def install_error_handlers(app: FastAPI) -> None:
    """Register exception handlers that return structured ErrorResponse bodies."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        status_code = exc.status_code
        return JSONResponse(
            status_code=status_code,
            content={
                "error": _STATUS_NAMES.get(status_code, "Error"),
                "detail": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                "trace_id": _get_trace_id(request),
                "status_code": status_code,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Unprocessable Entity",
                "detail": str(exc.errors()),
                "trace_id": _get_trace_id(request),
                "status_code": 422,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred",
                "trace_id": _get_trace_id(request),
                "status_code": 500,
            },
        )
