from typing import Any, Callable, Dict, List, Optional

from launch.pipeline.deploy import Deployment
from launch.pipeline.runtime import Runtime
from launch.pipeline.service import (
    SeqPipelineServiceDescription,
    SingleServiceDescription,
)


def make_service(
    service: Callable,
    runtime: Runtime,
    deploy: Deployment,
    init_kwargs: Optional[Dict[str, Any]] = None,
) -> SingleServiceDescription:
    """
    Create a structure that describes a single service.
    """
    return SingleServiceDescription(
        service=service,
        runtime=runtime,
        deploy=deploy,
        init_kwargs=init_kwargs,
    )


def make_sequential_pipeline(
    items: List[SingleServiceDescription],
) -> SeqPipelineServiceDescription:
    """
    Create a structure that describes a sequential pipeline.
    """
    return SeqPipelineServiceDescription(items=items)
