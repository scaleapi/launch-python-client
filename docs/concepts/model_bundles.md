# Model Bundles

Model Bundles are deployable models that can be used to make predictions. They
are created by packaging a model up into a deployable format. 

## Creating Model Bundles

There are two methods for creating model bundles:
[`create_model_bundle`](/api/client/#launch.client.LaunchClient.create_model_bundle)
and
[`create_model_bundle_from_dirs`](/api/client/#launch.client.LaunchClient.create_model_bundle_from_dirs).
The former directly pickles a user-specified `load_predict_fn`, a function which
loads the model and returns a `predict_fn`, a function which takes in a request.
The latter takes in directories containing a `load_predict_fn` and the
module path to the `load_predict_fn`.

=== "Creating a Model Bundle Directly"
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


    def my_model(x):
        return x * 2

    ENV_PARAMS = {
        "framework_type": "pytorch",
        "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
    }

    BUNDLE_PARAMS = {
        "model_bundle_name": "test-bundle",
        "model": my_model,
        "load_predict_fn": my_load_predict_fn,
        "env_params": ENV_PARAMS,
        "requirements": [],
        "request_schema": MyRequestSchema,
        "response_schema": MyResponseSchema,
    }

    client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
    client.create_model_bundle(**BUNDLE_PARAMS)
    ```

=== "Creating a Model Bundle from Directories"
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
    numpy
    sklearn
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
     
    ENV_PARAMS = {
        "framework_type": "pytorch",
        "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
    }

    BUNDLE_PARAMS = {
        "model_bundle_name": "test-bundle",
        "base_paths": [directory],
        "requirements_path": "requirements.txt",
        "env_params": ENV_PARAMS,
        "load_predict_fn": "predict.my_load_predict_fn",
        "load_model_fn_module_path": "model.my_load_model_fn",
        "request_schema": MyRequestSchema,
        "response_schema": MyResponseSchema,
    }

    client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
    client.create_model_bundle_from_dirs(**BUNDLE_PARAMS)

    # Clean up files from demo
    os.remove(model_filename)
    os.remove(predict_filename)
    os.rmdir(directory)
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


ENV_PARAMS = {
    "framework_type": "pytorch",
    "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
}

BUNDLE_PARAMS_SINGLE = {
    "model_bundle_name": "test-bundle-single",
    "load_predict_fn": my_load_predict_fn,
    "load_model_fn": my_load_model_fn,
    "env_params": ENV_PARAMS,
    "requirements": [],
    "request_schema": MyRequestSchema,
    "response_schema": MyResponseSchema,
    "app_config": {"mode": "single"},
}
BUNDLE_PARAMS_BATCHED = {
    "model_bundle_name": "test-bundle-batched",
    "load_predict_fn": my_load_predict_fn,
    "load_model_fn": my_load_model_fn,
    "env_params": ENV_PARAMS,
    "requirements": [],
    "request_schema": MyRequestSchema,
    "response_schema": MyResponseSchema,
    "app_config": {"mode": "batched"},
}

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
bundle_single = client.create_model_bundle(**BUNDLE_PARAMS_SINGLE)
bundle_batch = client.create_model_bundle(**BUNDLE_PARAMS_BATCHED)
```

## Updating Model Bundles

Model Bundles are immutable, meaning they cannot be edited once created.
However, it is possible to clone an existing model bundle with a new `app_config`
using
[`clone_model_bundle_with_changes`](/api/client/#launch.client.LaunchClient.clone_model_bundle_with_changes).

## Listing Model Bundles

To list all the model bundles you own, use
[`list_model_bundles`](/api/client/#launch.client.LaunchClient.list_model_bundles).
