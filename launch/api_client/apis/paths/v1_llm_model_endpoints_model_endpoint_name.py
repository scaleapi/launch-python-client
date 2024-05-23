from launch.api_client.paths.v1_llm_model_endpoints_model_endpoint_name.delete import (
    ApiFordelete,
)
from launch.api_client.paths.v1_llm_model_endpoints_model_endpoint_name.get import (
    ApiForget,
)
from launch.api_client.paths.v1_llm_model_endpoints_model_endpoint_name.put import (
    ApiForput,
)


class V1LlmModelEndpointsModelEndpointName(
    ApiForget,
    ApiForput,
    ApiFordelete,
):
    pass
