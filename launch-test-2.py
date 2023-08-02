from launch_internal import get_launch_client

gateway_endpoint = "launch-ian-macleod-98889099.ml-internal.scale.com"
# gateway_endpoint = 'https://launch.ml-serving-internal.scale.com'

worker_user = "64b6c01a46c52c03fb0d59b4"
user = "64c9ab182942a4a49da65e00"

client = get_launch_client(api_key=user, env="training", gateway_endpoint=gateway_endpoint)

response = client.download("fake_model", format="huggingface")
print(response)

response = client.download("llama-2-7b.2023-08-02-00-43-58", format="huggingface")
print(response.url)
