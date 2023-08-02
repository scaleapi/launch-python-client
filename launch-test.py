import os

from launch import EndpointRequest, LaunchClient

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"), endpoint="launch-ian-macleod-98889099.ml-internal.scale.com")

response = client.download("fake_model", format="huggingface")
print(response)

response = client.download("llama-2-7b.2023-08-02-00-43-58", format="huggingface")
print(response.url)
