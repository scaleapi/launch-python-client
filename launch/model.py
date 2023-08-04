from typing import List

from pydantic import BaseModel, Field


class ModelDownloadRequest(BaseModel):
    """Request object for downloading a model."""

    model_name: str = Field(..., description="Model name.")
    """Model name."""
    download_format: str = Field(..., description="Download format.")
    """Desired download format (default=huggingface)."""


class ModelDownloadResponse(BaseModel):
    """Response object for downloading a model."""

    urls: List[str] = Field(..., description="Model download urls.")
    """Model download urls."""
