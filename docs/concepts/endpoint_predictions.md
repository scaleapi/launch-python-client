# Endpoint Predictions

Once endpoints have been created, users can send tasks to them to make
predictions. The following code snippet shows how to send tasks to endpoints.


=== "Sending a Task to an Async Endpoint"
    ```py
    import os
    from launch import EndpointRequest, LaunchClient

    client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
    endpoint = client.get_model_endpoint("demo-endpoint-async")
    future = endpoint.predict(request=EndpointRequest(args={"x": 2, "y": "hello"}))
    response = future.get()
    print(response)
    ```

=== "Sending a Task to a Sync Endpoint"
    ```py
    import os
    from launch import EndpointRequest, LaunchClient

    client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
    endpoint = client.get_model_endpoint("demo-endpoint-sync")
    response = endpoint.predict(request=EndpointRequest(args={"x": 2, "y": "hello"}))
    print(response)
    ```

=== "Sending a Task to a Streaming Endpoint"
    ```py
    import os
    from launch import EndpointRequest, LaunchClient

    client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
    endpoint = client.get_model_endpoint("demo-endpoint-streaming")
    response = endpoint.predict(request=EndpointRequest(args={"x": 2, "y": "hello"}))
    print(response)
    ```

::: launch.model_endpoint.EndpointRequest
::: launch.model_endpoint.EndpointResponseFuture
::: launch.model_endpoint.EndpointResponse
::: launch.model_endpoint.EndpointResponseStream
