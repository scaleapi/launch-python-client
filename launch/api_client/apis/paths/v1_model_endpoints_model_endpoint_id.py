from launch.api_client.paths.v1_model_endpoints_model_endpoint_id.delete import (
    ApiFordelete,
)
from launch.api_client.paths.v1_model_endpoints_model_endpoint_id.get import (
    ApiForget,
)
from launch.api_client.paths.v1_model_endpoints_model_endpoint_id.put import (
    ApiForput,
)


class V1ModelEndpointsModelEndpointId(
    ApiForget,
    ApiForput,
    ApiFordelete,
):
    pass
