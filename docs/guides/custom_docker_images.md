# Custom docker images

!!! Warning
    This feature is currently in beta, and the API is likely to change. Please contact us if you are interested
    in using this feature.

If you need more customization that what cloudpickle or zip artifacts can offer, or if you just already have a pre-built
docker image, then you can create a Model Bundle with that docker image. You will need to modify your image to run a
web server that exposes HTTP port 5005.

In our example below, we assume that you have some existing Python function `my_inference_fn` that can be imported.
If you need to invoke some other binary (e.g. a custom C++ binary), then you can shell out to the OS to call that binary;
subsequent versions of this document will have native examples for non-Python binaries.

For choice of web server, we recommend [FastAPI](https://fastapi.tiangolo.com/lo/) due to its speed and ergonomics.
Any web server would work, although we give examples with FastAPI. 

## Step 1: Install FastAPI

You can add `fastapi` to the `requirements.txt` file that gets installed as part of your Dockerfile. Alternatively,
you can add `pip install fastapi` to the Dockerfile directly.

## Step 2: Set up a web server application

Inside your project workspace, create a `server.py` file with these contents:

```py
from fastapi import FastAPI

from pydantic import BaseModel

app = FastAPI()

class MyRequestSchema(BaseModel):
    inputs: str


class MyResponseSchema(BaseModel):
    response: str

def my_inference_fn(req: MyRequestSchema) -> MyResponseSchema:
    # This is an example inference function - you can instead import a function from your own codebase,
    # or shell out to the OS, etc.
    resp = req.inputs + "_hello"
    return MyResponseSchema(response=resp)

@app.get("/predict")
async def predict(request: MyRequestSchema) -> MyResponseSchema:
    response = my_inference_fn(request)
    return response

@app.get("/readyz")
def readyz():
    return "ok"
```

## Step 3: Rebuild and push your image

Build your updated Dockerfile and push the image to a location that is accessible by Scale. For instance, if you are
using AWS ECR, please make sure that the necessary cross-account permissions allow Scale to pull your docker image.

## Step 4: Deploy!

Now you can upload your docker image as a Model Bundle, and then create a Model Endpoint referencing that Model Bundle.


```py
import os

from launch import LaunchClient

from server import MyRequestSchema, MyResponseSchema  # Defined as part of your server.py

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))

model_bundle_name = "my_bundle_name"

client.create_model_bundle_from_runnable_image_v2(
    model_bundle_name=model_bundle_name,
    request_schema=MyRequestSchema,
    response_schema=MyResponseSchema,
    repository="$YOUR_ECR_REPO",
    tag="$YOUR_IMAGE_TAG",
    command=[
        "dumb-init",
        "--",
        "uvicorn",
        "/path/in/docker/image/to/server.py",
        "--port",
        "5005",
        "--host",
        "::1",
    ],
    readiness_initial_delay_seconds=120,
    env={},
)

client.create_model_endpoint(
    endpoint_name=f"endpoint-{model_bundle_name}",
    model_bundle=model_bundle_name,
    endpoint_type="async",
    min_workers=0,
    max_workers=1,
    per_worker=1,
    memory="30Gi",
    storage="40Gi",
    cpus=4,
    gpus=1,
    gpu_type="nvidia-ampere-a10",
    update_if_exists=True,
)
```
