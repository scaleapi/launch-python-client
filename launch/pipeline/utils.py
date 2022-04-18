from typing import Any, Callable, Dict, List, Optional

from launch.pipeline.deployment import Deployment
from launch.pipeline.runtime import Runtime
from launch.pipeline.service import (
    SequentialPipelineDescription,
    SingleServiceDescription,
)


def make_service(
    service: Callable,
    runtime: Runtime,
    deployment: Deployment,
    init_kwargs: Optional[Dict[str, Any]] = None,
) -> SingleServiceDescription:
    """
    Create a structure that describes a single service.
    """
    return SingleServiceDescription(
        service=service,
        runtime=runtime,
        deployment=deployment,
        init_kwargs=init_kwargs,
    )


def make_sequential_pipeline(
    items: List[SingleServiceDescription],
) -> SequentialPipelineDescription:
    """
    Create a structure that describes a sequential pipeline.
    """
    return SequentialPipelineDescription(items=items)
