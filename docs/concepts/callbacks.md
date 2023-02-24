# Callbacks

Async model endpoints can be configured to send callbacks to a user-defined
callback URL. Callbacks are sent as HTTP POST requests with a JSON body. The
following code snippet shows how to create an async model endpoint with a
callback URL.

To configure an async endpoint to send callbacks, set the `post_inference_hooks`
field to include
[`launch.PostInferenceHooks.CALLBACK`](/api/hooks/#launch.hooks.PostInferenceHooks).
A callback URL also needs to be specified, and it can be configured as a default
using the `default_callback_url` argument to
[`launch.LaunchClient.create_model_endpoint`](/api/client/#launch.LaunchClient.create_model_endpoint).
or as a per-task override using the `callback_url` field of
[`launch.EndpointRequest`](/api/model_endpoints/#launch.model_predictions.EndpointRequest).

!!! Note
    Callbacks will not be sent if the endpoint does not have any post-inference
    hooks specified, even if a `default_callback_url` is provided to the endpoint
    creation method or if the prediction request has a `callback_url` override.


```py title="Creating an Async Model Endpoint with a Callback URL"
import os
from launch import EndpointRequest, LaunchClient, PostInferenceHooks

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
endpoint = client.create_model_endpoint(
    endpoint_name="demo-endpoint-callback",
    model_bundle="test-bundle",
    cpus=1,
    min_workers=1,
    endpoint_type="async",
    update_if_exists=True,
    labels={
        "team": "MY_TEAM",
        "product": "MY_PRODUCT"
    },
    post_inference_hooks=[PostInferenceHooks.CALLBACK],
    default_callback_url="https://example.com",
)

future_default = endpoint.predict(request=EndpointRequest(args={"x": 2, "y": "hello"}))
"""
A callback is sent to https://example.com with the following JSON body:
{
    "task_id": "THE_TASK_ID",
    "result": 7
}
"""


future_custom_callback_url = endpoint.predict(
    request=EndpointRequest(
        args={"x": 3, "y": "hello"}, callback_url="https://example.com/custom"
    ),
)

"""
A callback is sent to https://example.com/custom with the following JSON body:
{
    "task_id": "THE_TASK_ID",
    "result": 8
}
"""
```
