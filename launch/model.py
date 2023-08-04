from typing import List

from pydantic import BaseModel, Field


class ModelDownloadRequest(BaseModel):
    """Request object for downloading a model."""

    owner: str = Field(..., description="Owner ID.")
    """ID of the model owner."""
    model_name: str = Field(..., description="Model name.")
    """Model name."""


class ModelDownloadResponse(BaseModel):
    """Response object for downloading a model."""

    urls: List[str] = Field(..., description="Model download urls.")
    """Model download urls."""
