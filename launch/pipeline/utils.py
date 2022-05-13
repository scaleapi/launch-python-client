from typing import Any, Callable, Dict, List

from launch.pipeline.deployment import Deployment
from launch.pipeline.runtime import Runtime
from launch.pipeline.service import (
    SequentialPipelineDescription,
    SingleServiceDescription,
    ServiceDescription,
)


def find_all_sub_service_descriptions(
    target_service: ServiceDescription,
) -> List[ServiceDescription]:
    """
    Find all the services involved in the pipeline.
    """
    service_descriptions = [target_service]
    if isinstance(target_service, SequentialPipelineDescription):
        for sub_service_description in target_service.items:
            service_descriptions.extend(
                find_all_sub_service_descriptions(sub_service_description)
            )
    return service_descriptions


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
    runtime: Runtime,
    deployment: Deployment,
) -> SequentialPipelineDescription:
    """
    Create a structure that describes a sequential pipeline.
    """
    return SequentialPipelineDescription(
        items=items,
        runtime=runtime,
        deployment=deployment,
    )
