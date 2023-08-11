from typing import Dict

from pydantic import BaseModel, Field


class ModelDownloadRequest(BaseModel):
    """Request object for downloading a model."""

    model_name: str = Field(..., description="Model name.")
    """Model name."""
    download_format: str = Field(..., description="Download format.")
    """Desired download format (default=huggingface)."""


class ModelDownloadResponse(BaseModel):
    """Response object for downloading a model."""

    urls: Dict[str, str] = Field(..., description="Dictionary of model file name, model download url pairs.")
    """Model download urls."""
