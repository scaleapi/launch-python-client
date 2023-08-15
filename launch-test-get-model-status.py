from launch_internal import get_launch_client

gateway_endpoint = "https://model-engine-ian-macleod-0090009890.ml-internal.scale.com"
worker_user = "64b6c01a46c52c03fb0d59b4"

client = get_launch_client(api_key=worker_user, env="training", gateway_endpoint=gateway_endpoint)

response = client.list_llm_model_endpoints()
print(response)
