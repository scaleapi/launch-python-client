from typing import Any, Callable, Dict, List

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
    **kwargs: Dict[str, Any],
) -> SingleServiceDescription:
    """
    Create a structure that describes a single service.
    """
    return SingleServiceDescription(
        service=service,
        runtime=runtime,
        deployment=deployment,
        **kwargs,
    )


def make_sequential_pipeline(
    items: List[SingleServiceDescription],
) -> SequentialPipelineDescription:
    """
    Create a structure that describes a sequential pipeline.
    """
    return SequentialPipelineDescription(items=items)
