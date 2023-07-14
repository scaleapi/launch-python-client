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


class CreateFineTuneJobResponse(BaseModel):
    fine_tune_id: str
    """ID of the created fine-tuning job"""


class GetFineTuneJobResponse(BaseModel):
    fine_tune_id: str
    """ID of the requested job"""
    status: BatchJobStatus
    """Status of the requested job"""


class ListFineTuneJobResponse(BaseModel):
    jobs: List[GetFineTuneJobResponse]
    """List of fine-tuning jobs and their statuses"""


class CancelFineTuneJobResponse(BaseModel):
    success: bool
    """Whether cancellation was successful"""
