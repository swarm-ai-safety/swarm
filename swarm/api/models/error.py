"""Structured error response model."""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Consistent error response format for all API errors."""

    error: str
    detail: str
    trace_id: str
    status_code: int
