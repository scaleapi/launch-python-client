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
[`launch.LaunchClient.create_model_endpoint`](/api/client/#launch.LaunchClient.create_model_endpoint)
or as a per-task override using the `callback_url` field of
[`launch.EndpointRequest`](/api/model_endpoints/#launch.model_predictions.EndpointRequest).

!!! Note
    Callbacks will not be sent if the endpoint does not have any post-inference
    hooks specified, even if a `default_callback_url` is provided to the endpoint
    creation method or if the prediction request has a `callback_url` override.


```py title="Creating an Async Model Endpoint with a Callback URL"  hl_lines="17-18 37"
import os
import time
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
        "product": "MY_PRODUCT",
    },
    post_inference_hooks=[PostInferenceHooks.CALLBACK],
    default_callback_url="https://example.com",
)

while endpoint.status() != "READY":
    time.sleep(10)

future_default = endpoint.predict(
    request=EndpointRequest(args={"x": 2, "y": "hello"})
)
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

## Authentication for callbacks

!!! Warning
    This feature is currently in beta, and the API is likely to change.

Callbacks can be authenticated using shared authentication headers. To enable authentication,
set either `default_callback_auth_kind` when creating the endpoint or `callback_auth_kind`
when making a prediction request.

Currently, the supported authentication methods are `basic` and `mtls`. If `basic` is used,
then the `default_callback_auth_username` and `default_callback_auth_password` fields must be
specified when creating the endpoint, or the `callback_auth_username` and `callback_auth_password`
fields must be specified when making a prediction request. If `mtls` is used, then the 
same is true for the `default_callback_auth_cert` and `default_callback_auth_key` fields,
or the `callback_auth_cert` and `callback_auth_key` fields.

```py title="Creating an Async Model Endpoint with custom Callback auth"  hl_lines="18-21 37-39"
import os
import time
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
        "product": "MY_PRODUCT",
    },
    post_inference_hooks=[PostInferenceHooks.CALLBACK],
    default_callback_url="https://example.com",
    default_callback_auth_kind="basic",
    default_callback_auth_username="user",
    default_callback_auth_password="password",
)

while endpoint.status() != "READY":
    time.sleep(10)

future_default = endpoint.predict(
    request=EndpointRequest(args={"x": 2, "y": "hello"})
)
"""
A callback is sent to https://example.com with ("user", "password") as the basic auth.
"""

future_custom_callback_auth = endpoint.predict(
    request=EndpointRequest(
        args={"x": 3, "y": "hello"},
        callback_auth_kind="mtls", 
        callback_auth_cert="cert", 
        callback_auth_key="key",
    ),
)

"""
A callback is sent with mTLS authentication.
"""
```
