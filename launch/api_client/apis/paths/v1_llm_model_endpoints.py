from launch.api_client.paths.v1_llm_model_endpoints.get import ApiForget
from launch.api_client.paths.v1_llm_model_endpoints.post import ApiForpost


class V1LlmModelEndpoints(
    ApiForget,
    ApiForpost,
):
    pass
