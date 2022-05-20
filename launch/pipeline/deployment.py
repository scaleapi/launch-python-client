from dataclasses import dataclass
from typing import Optional


@dataclass
class Deployment:
    """Base deployment class"""

    cpus: int = 1
    gpus: int = 0
    gpu_type: Optional[str] = None
    memory: str = "4Gi"
    min_workers: int = 0
    max_workers: int = 1
    per_worker: int = 1
