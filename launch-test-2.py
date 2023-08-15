from launch_internal import get_launch_client

gateway_endpoint = "https://model-engine-ian-macleod-80000.ml-internal.scale.com"

worker_user = "64b6c01a46c52c03fb0d59b4"

client = get_launch_client(api_key=worker_user, env="training", gateway_endpoint=gateway_endpoint)

response = client.download_model_weights(
    "llama-2-7b.ray-test-fine-deploy.2023-08-04-17-40-13", download_format="huggingface"
)
print(response.urls)
