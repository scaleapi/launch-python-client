from enum import Enum
from typing import List

from pydantic import BaseModel


class BatchJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    UNDEFINED = "UNDEFINED"
    TIMEOUT = "TIMEOUT"


class CreateFineTuneResponse(BaseModel):
    fine_tune_id: str
    """ID of the created fine-tuning job"""


class GetFineTuneResponse(BaseModel):
    fine_tune_id: str
    """ID of the requested job"""
    status: BatchJobStatus
    """Status of the requested job"""


class ListFineTunesResponse(BaseModel):
    jobs: List[GetFineTuneResponse]
    """List of fine-tuning jobs and their statuses"""


class CancelFineTuneResponse(BaseModel):
    success: bool
    """Whether cancellation was successful"""
