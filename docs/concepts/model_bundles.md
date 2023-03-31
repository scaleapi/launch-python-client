# Model Bundles

Model Bundles are deployable models that can be used to make predictions. They
are created by packaging a model up into a deployable format. 

## Creating Model Bundles

There are three methods for creating model bundles:
[`create_model_bundle_from_callable_v2`](/api/client/#launch.client.LaunchClient.create_model_bundle_from_callable_v2),
[`create_model_bundle_from_dirs_v2`](/api/client/#launch.client.LaunchClient.create_model_bundle_from_dirs_v2),
and
[`create_model_bundle_from_runnable_image_v2`](/api/client/#launch.client.LaunchClient.create_model_bundle_from_runnable_image_v2).
The first directly pickles a user-specified `load_predict_fn`, a function which
loads the model and returns a `predict_fn`, a function which takes in a request.
The second takes in directories containing a `load_predict_fn` and the
module path to the `load_predict_fn`.
The third takes a Docker image and a command that starts a process listening for
requests at port 5005 using HTTP and exposes `POST /predict` and
`GET /readyz` endpoints.

Each of these modes of creating a model bundle is called a "Flavor".

!!! info
    # Choosing the right model bundle flavor

    Here are some tips for how to choose between the different flavors of ModelBundle:

    A `CloudpickleArtifactFlavor` (creating from callable) is good if:

    * You are creating the model bundle from a Jupyter notebook.
    * The model bundle is small without too many dependencies.

    A `ZipArtifactFlavor` (creating from directories) is good if:

    * You have a relatively constant set of dependencies.
    * You have a lot of custom code that you want to include in the model bundle.
    * You do not want to build a web server and Docker image to serve your model.

    A `RunnableImageFlavor` (creating from runnable image) is good if:

    * You have a lot of dependencies.
    * You have a lot of custom code that you want to include in the model bundle.
    * You are comfortable with building a web server and Docker image to serve your model.


    A `TritonEnhancedRunnableImageFlavor` (a runnable image variant) is good if:

    * You want to use a `RunnableImageFlavor`
    * You also want to use [NVidia's `tritonserver`](https://developer.nvidia.com/nvidia-triton-inference-server) to accelerate model inference


=== "Creating From Callables"
    ```py
    import os
    from pydantic import BaseModel
    from launch import LaunchClient


    class MyRequestSchema(BaseModel):
        x: int
        y: str

    class MyResponseSchema(BaseModel):
        __root__: int


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
        "load_model_fn": my_load_model_fn,
        "load_predict_fn": my_load_predict_fn,
        "request_schema": MyRequestSchema,
        "response_schema": MyResponseSchema,
        "requirements": ["pytest==7.2.1", "numpy"],  # list your requirements here
        "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
    }

    client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
    client.create_model_bundle_from_callable_v2(**BUNDLE_PARAMS)
    ```

=== "Creating From Directories"
    ```py
    import os
    import tempfile
    from pydantic import BaseModel
    from launch import LaunchClient

    directory = tempfile.mkdtemp()
    model_filename = os.path.join(directory, "model.py")
    with open(model_filename, "w") as f:
        f.write("""
    def my_load_model_fn():
        def my_model(x):
            return x * 2

        return my_model
    """)

    predict_filename = os.path.join(directory, "predict.py")
    with open(predict_filename, "w") as f:
        f.write("""
    def my_load_predict_fn(model):
        def returns_model_of_x_plus_len_of_y(x: int, y: str) -> int:
            assert isinstance(x, int) and isinstance(y, str)
            return model(x) + len(y)

        return returns_model_of_x_plus_len_of_y
    """)

    requirements_filename = os.path.join(directory, "requirements.txt")
    with open(predict_filename, "w") as f:
        f.write("""
    pytest==7.2.1
    numpy
    """
        )
     
    """
    The directory structure should now look like

    directory/
        model.py
        predict.py
        requirements.txt
    """


    class MyRequestSchema(BaseModel):
        x: int
        y: str

    class MyResponseSchema(BaseModel):
        __root__: int
     
    BUNDLE_PARAMS = {
        "model_bundle_name": "test-bundle",
        "base_paths": [directory],
        "load_predict_fn_module_path": "predict.my_load_predict_fn",
        "load_model_fn_module_path": "model.my_load_model_fn",
        "request_schema": MyRequestSchema,
        "response_schema": MyResponseSchema,
        "requirements_path": "requirements.txt",
        "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
    }

    client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
    client.create_model_bundle_from_dirs_v2(**BUNDLE_PARAMS)

    # Clean up files from demo
    os.remove(model_filename)
    os.remove(predict_filename)
    os.rmdir(directory)
    ```

=== "Creating From a Runnable Image"
    ```py
    import os
    from pydantic import BaseModel
    from launch import LaunchClient


    class MyRequestSchema(BaseModel):
        x: int
        y: str

    class MyResponseSchema(BaseModel):
        __root__: int


    BUNDLE_PARAMS = {
        "model_bundle_name": "test-bundle",
        "request_schema": MyRequestSchema,
        "response_schema": MyResponseSchema,
        "repository": "...",
        "tag": "...",
        "command": [
            ...
        ],
        "env": {
            "TEST_KEY": "test_value",
        },
        "readiness_initial_delay_seconds": 30,

    }

    client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
    client.create_model_bundle_from_runnable_image_v2(**BUNDLE_PARAMS)
    ```


=== "Creating From a Triton Enhanced Runnable Image"
    ```py
    import os
    from pydantic import BaseModel
    from launch import LaunchClient


    class MyRequestSchema(BaseModel):
        x: int
        y: str

    class MyResponseSchema(BaseModel):
        __root__: int


    BUNDLE_PARAMS = {
        "model_bundle_name": "test-bundle",
        "request_schema": MyRequestSchema,
        "response_schema": MyResponseSchema,
        "repository": "...",
        "tag": "...",
        "command": [
            ...
        ],
        "env": {
            "TEST_KEY": "test_value",
        },
        "readiness_initial_delay_seconds": 30,
        "triton_model_repository": "...",
        "triton_model_replicas": {"": ""},
        "triton_num_cpu": 4.0,
        "triton_commit_tag": "",
        "triton_storage": "",
        "triton_memory": "",
        "triton_readiness_initial_delay_seconds": 300,
    }

    client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
    client.create_model_bundle_from_triton_enhanced_runnable_image_v2(**BUNDLE_PARAMS)
    ```


## Configuring Model Bundles

The `app_config` field of a model bundle is a dictionary that can be used to
configure the model bundle. If specified, the `app_config` is passed to the
`load_predict_fn` when the model bundle is deployed, alongside the `model`. This
can allow for more code reuse between multiple bundles that perform similar
tasks.

```py title="Creating Model Bundles with app_config"
import os
from launch import LaunchClient
from pydantic import BaseModel
from typing import List, Union
from typing_extensions import Literal


class MyRequestSchemaSingle(BaseModel):
    kind: Literal['single']
    x: int
    y: str

class MyRequestSchemaBatched(BaseModel):
    kind: Literal['batched']
    x: List[int]
    y: List[str]

class MyRequestSchema(BaseModel):
    __root__: Union[MyRequestSchemaSingle, MyRequestSchemaBatched]

class MyResponseSchema(BaseModel):
    __root__: Union[int, List[int]]


def my_load_predict_fn(app_config, model):
    def returns_model_of_x_plus_len_of_y(x: Union[int, List[int]], y: Union[str, List[str]]) -> Union[int, List[int]]:
        """MyRequestSchema -> MyResponseSchema"""
        if app_config["mode"] == "single":
            assert isinstance(x, int) and isinstance(y, str)
            return model(x) + len(y)
 
        result = []
        for x_i, y_i in zip(x, y):
            result.append(model(x_i) + len(y_i))
        return result

    return returns_model_of_x_plus_len_of_y


def my_load_model_fn(app_config):
    def my_model_single(x: int):
        return x * 2
 
    def my_model_batched(x: List[int]):
        return [my_model_single(x_i) for x_i in x]
    
    if app_config["mode"] == "single":
        return my_model_single
    
    return my_model_batched


BUNDLE_PARAMS_SINGLE = {
    "model_bundle_name": "test-bundle-single",
    "load_predict_fn": my_load_predict_fn,
    "load_model_fn": my_load_model_fn,
    "requirements": ["pytest==7.2.1", "numpy"],
    "request_schema": MyRequestSchema,
    "response_schema": MyResponseSchema,
    "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
    "app_config": {"mode": "single"},
}
BUNDLE_PARAMS_BATCHED = {
    "model_bundle_name": "test-bundle-batched",
    "load_predict_fn": my_load_predict_fn,
    "load_model_fn": my_load_model_fn,
    "requirements": ["pytest==7.2.1", "numpy"],
    "request_schema": MyRequestSchema,
    "response_schema": MyResponseSchema,
    "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
    "app_config": {"mode": "batched"},
}

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
bundle_single = client.create_model_bundle_from_callable_v2(**BUNDLE_PARAMS_SINGLE)
bundle_batch = client.create_model_bundle_from_callable_v2(**BUNDLE_PARAMS_BATCHED)
```

## Updating Model Bundles

Model Bundles are immutable, meaning they cannot be edited once created.
However, it is possible to clone an existing model bundle with a new `app_config`
using
[`clone_model_bundle_with_changes_v2`](/api/client/#launch.client.LaunchClient.clone_model_bundle_with_changes_v2).

## Listing Model Bundles

To list all the model bundles you own, use
[`list_model_bundles_v2`](/api/client/#launch.client.LaunchClient.list_model_bundles_v2).
