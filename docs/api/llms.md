# LLM APIs

We provide some APIs to conveniently create, list and inference with LLMs. Under the hood they are Launch model endpoints.

## Example

```py title="LLM APIs Usage"
import os

from rich import print

from launch import LaunchClient
from launch.api_client.model.llm_inference_framework import (
    LLMInferenceFramework,
)
from launch.api_client.model.llm_source import LLMSource

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"), endpoint=os.getenv("LAUNCH_ENDPOINT"))

endpoints = client.list_llm_model_endpoints()

print(endpoints)

endpoint_name = "test-flan-t5-xxl"
client.create_llm_model_endpoint(
    endpoint_name=endpoint_name,
    model_name="flan-t5-xxl",
    source=LLMSource.HUGGING_FACE,
    inference_framework=LLMInferenceFramework.DEEPSPEED,
    inference_framework_image_tag=os.getenv("INFERENCE_FRAMEWORK_IMAGE_TAG"),
    num_shards=4,
    min_workers=1,
    max_workers=1,
    gpus=4,
    endpoint_type="sync",
)

# Wait for the endpoint to be ready

output = client.completions_sync(endpoint_name, prompt="What is Deep Learning?", max_new_tokens=10, temperature=0)
print(output)
```
