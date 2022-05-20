import logging
from typing import Any, Callable, Dict, List, Optional

from launch.model_endpoint import Endpoint
from launch.pipeline.deployment import Deployment
from launch.pipeline.runtime import Runtime
from launch.pipeline.service import (
    SequentialPipelineDescription,
    ServiceDescription,
    SingleServiceDescription,
)

logger = logging.getLogger(__name__)
logging.basicConfig()


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
    env_params: Dict[str, str],
    **kwargs: Dict[str, Any],
) -> SingleServiceDescription:
    """
    Create a structure that describes a single service.
    """
    return SingleServiceDescription(
        service=service,
        runtime=runtime,
        deployment=deployment,
        env_params=env_params,
        **kwargs,
    )


def make_sequential_pipeline(
    items: List[SingleServiceDescription],
    runtime: Runtime,
    deployment: Deployment,
    env_params: Dict[str, str],
) -> SequentialPipelineDescription:
    """
    Create a structure that describes a sequential pipeline.
    """
    return SequentialPipelineDescription(
        items=items,
        runtime=runtime,
        deployment=deployment,
        env_params=env_params,
    )


def deploy_service(
    client,
    model_bundle_name: str,
    endpoint_name: str,
    service_description: ServiceDescription,
    bundle_url_prefix: Optional[str] = None,
    globals_copy: Optional[Dict[str, Any]] = None,
) -> Optional[Endpoint]:
    """
    Deploy a service represented by a Service Description object such as a Step or a Pipeline.

    Parameters:
        client: Launch API client.
        model_bundle_name: Name of model bundle you want to create. This acts as a unique identifier.
        endpoint_name: Name of model endpoint. Must be unique.
        service_description: A step or a Pipeline.
        bundle_url_prefix: Only for self-hosted mode. Desired location of bundle.
        globals_copy: Dictionary of the global symbol table. Normally provided by `globals()` built-in function.

    Returns:
         A Endpoint object that can be used to make requests to the endpoint.

    """
    sub_service_descriptions = find_all_sub_service_descriptions(
        service_description
    )
    # Fill `child_fn_info` needed for service discovery
    child_fn_info = {}
    for sub_service_description in sub_service_descriptions:
        # If this is the root service target.
        if (
            service_description.service_name
            == sub_service_description.service_name
        ):
            sub_service_description.service_name = model_bundle_name

        child_fn_info[sub_service_description.service_name] = dict(
            remote=True,
            endpoint_type=sub_service_description.runtime.name.lower(),
            destination="undefined",
        )

    res_endpoint = None
    for sub_service_description in sub_service_descriptions:
        service_name = sub_service_description.service_name

        if bundle_url_prefix:
            bundle_url = f"{bundle_url_prefix}/{service_name}.pkl"
        else:
            bundle_url = None

        model_bundle = client.create_model_bundle(
            service_name,
            predict_fn_or_cls=sub_service_description.service,
            bundle_url=bundle_url,
            env_params=sub_service_description.env_params,
            globals_copy=globals_copy,
        )

        logger.info("Bundle %s created", service_name)

        endpoint = client.create_model_endpoint(
            endpoint_name=service_name,
            model_bundle=model_bundle,
            cpus=service_description.deployment.cpus,
            memory=service_description.deployment.memory,
            gpus=service_description.deployment.gpus,
            min_workers=service_description.deployment.min_workers,
            max_workers=service_description.deployment.max_workers,
            per_worker=service_description.deployment.per_worker,
            gpu_type=service_description.deployment.gpu_type,
            endpoint_type=service_description.runtime.name.lower(),
            update_if_exists=False,
            child_fn_info=child_fn_info,
        )
        logger.info("Endpoint %s created", service_name)

        if (
            service_description.service_name
            == sub_service_description.service_name
        ):
            res_endpoint = endpoint
    return res_endpoint
