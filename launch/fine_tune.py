from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class BatchJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    UNDEFINED = "UNDEFINED"
    TIMEOUT = "TIMEOUT"


class LLMFineTuneEvent(BaseModel):
    timestamp: Optional[float] = None
    message: str
    level: str


class CreateFineTuneResponse(BaseModel):
    id: str
    """ID of the created fine-tuning job"""


class GetFineTuneResponse(BaseModel):
    id: str
    """ID of the requested job"""
    fine_tuned_model: Optional[str] = None
    """
    Name of the resulting fine-tuned model. This can be plugged into the
    Completion API ones the fine-tune is complete
    """
    status: BatchJobStatus
    """Status of the requested job"""


class ListFineTunesResponse(BaseModel):
    jobs: List[GetFineTuneResponse]
    """List of fine-tuning jobs and their statuses"""


class CancelFineTuneResponse(BaseModel):
    success: bool
    """Whether cancellation was successful"""


class GetFineTuneEventsResponse(BaseModel):
    events: List[LLMFineTuneEvent]
