# Scale Launch

[![CI](https://circleci.com/gh/scaleapi/launch-python-client.svg)](https://circleci.com/gh/scaleapi/launch-python-client)
[![pypi](https://img.shields.io/pypi/v/scale-launch.svg)](https://pypi.python.org/pypi/scale-launch)

Simple, scalable, and high performance ML service deployment in python.

## Example

```py title="Launch Usage"
import os
import time
from launch import LaunchClient
from launch import EndpointRequest
from pydantic import BaseModel, RootModel
from rich import print


class MyRequestSchema(BaseModel):
    x: int
    y: str

class MyResponseSchema(RootModel):
    root: int


def my_load_predict_fn(model):
    def returns_model_of_x_plus_len_of_y(x: int, y: str) -> int:
        """MyRequestSchema -> MyResponseSchema"""
        assert isinstance(x, int) and isinstance(y, str)
        return model(x) + len(y)

    return returns_model_of_x_plus_len_of_y


def my_load_model_fn():
    def my_model(x):
        return x * 2
  
    return my_model

BUNDLE_PARAMS = {
    "model_bundle_name": "test-bundle",
    "load_predict_fn": my_load_predict_fn,
    "load_model_fn": my_load_model_fn,
    "request_schema": MyRequestSchema,
    "response_schema": MyResponseSchema,
    "requirements": ["pytest==7.2.1", "numpy"],  # list your requirements here
    "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
}

ENDPOINT_PARAMS = {
    "endpoint_name": "demo-endpoint",
    "model_bundle": "test-bundle",
    "cpus": 1,
    "min_workers": 0,
    "endpoint_type": "async",
    "update_if_exists": True,
    "labels": {
        "team": "MY_TEAM",
        "product": "launch",
    }
}

def predict_on_endpoint(request: MyRequestSchema) -> MyResponseSchema:
    # Wait for the endpoint to be ready first before submitting a task
    endpoint = client.get_model_endpoint(endpoint_name="demo-endpoint")
    while endpoint.status() != "READY":
        time.sleep(10)

    endpoint_request = EndpointRequest(args=request.dict(), return_pickled=False)

    future = endpoint.predict(request=endpoint_request)
    raw_response = future.get()

    response = MyResponseSchema.parse_raw(raw_response.result)
    return response


client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))

client.create_model_bundle_from_callable_v2(**BUNDLE_PARAMS)
endpoint = client.create_model_endpoint(**ENDPOINT_PARAMS)

request = MyRequestSchema(x=5, y="hello")
response = predict_on_endpoint(request)
print(response)
"""
MyResponseSchema(root=10)
"""
```

What's going on here:

* First we use [`pydantic`](https://github.com/pydantic/pydantic) to define our request and response
  schemas, `MyRequestSchema` and `MyResponseSchema`. These schemas are used to generate the API
  documentation for our models.
* Next we define the the `model` and the `load_predict_fn`, which tells Launch
  how to load our model and how to make predictions with it. In this case,
  we're just returning a function that adds the length of the string `y` to 
  `model(x)`, where `model` doubles the integer `x`.
* We then define the model bundle by specifying the `load_predict_fn`, the `request_schema`, and the
  `response_schema`. We also specify the `env_params`, which tell Launch environment settings like 
  the base image to use. In this case, we're using a PyTorch image.
* Next, we create the model endpoint, which is the API that we'll use to make predictions. We
  specify the `model_bundle` that we created above, and we specify the `endpoint_type`, which tells
  Launch whether to use a synchronous or asynchronous endpoint. In this case, we're using an
  asynchronous endpoint, which means that we can make predictions and return immediately with a
  `future` object. We can then use the `future` object to get the prediction result later.
* Finally, we make a prediction by calling `predict_on_endpoint` with a `MyRequestSchema` object.
  This function first waits for the endpoint to be ready, then it submits a prediction request to
  the endpoint. It then waits for the prediction result and returns it.

Notice that we specified `min_workers=0`, meaning that the endpoint will scale down to 0 workers
when it's not being used.

## Installation

To use Scale Launch, first install it using `pip`:

```commandline title="Installation"
pip install -U scale-launch
```
