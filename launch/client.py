import inspect  # pylint: disable=C0302
import json
import logging
import os
import shutil
import tempfile
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from zipfile import ZipFile

import cloudpickle
import requests
import yaml
from deprecation import deprecated
from frozendict import frozendict
from pydantic import BaseModel
from typing_extensions import Literal

from launch.api_client import ApiClient, Configuration
from launch.api_client.apis.tags.default_api import DefaultApi
from launch.api_client.model.callback_auth import CallbackAuth
from launch.api_client.model.clone_model_bundle_v1_request import (
    CloneModelBundleV1Request,
)
from launch.api_client.model.clone_model_bundle_v2_request import (
    CloneModelBundleV2Request,
)
from launch.api_client.model.cloudpickle_artifact_flavor import (
    CloudpickleArtifactFlavor,
)
from launch.api_client.model.completion_sync_v1_request import (
    CompletionSyncV1Request,
)
from launch.api_client.model.completion_sync_v1_response import (
    CompletionSyncV1Response,
)
from launch.api_client.model.create_batch_job_v1_request import (
    CreateBatchJobV1Request,
)
from launch.api_client.model.create_docker_image_batch_job_bundle_v1_request import (
    CreateDockerImageBatchJobBundleV1Request,
)
from launch.api_client.model.create_docker_image_batch_job_v1_request import (
    CreateDockerImageBatchJobV1Request,
)
from launch.api_client.model.create_llm_model_endpoint_v1_request import (
    CreateLLMModelEndpointV1Request,
)
from launch.api_client.model.create_model_bundle_v1_request import (
    CreateModelBundleV1Request,
)
from launch.api_client.model.create_model_bundle_v2_request import (
    CreateModelBundleV2Request,
)
from launch.api_client.model.create_model_endpoint_v1_request import (
    CreateModelEndpointV1Request,
)
from launch.api_client.model.custom_framework import CustomFramework
from launch.api_client.model.endpoint_predict_v1_request import (
    EndpointPredictV1Request,
)
from launch.api_client.model.gpu_type import GpuType
from launch.api_client.model.llm_inference_framework import (
    LLMInferenceFramework,
)
from launch.api_client.model.llm_source import LLMSource
from launch.api_client.model.model_bundle_environment_params import (
    ModelBundleEnvironmentParams,
)
from launch.api_client.model.model_bundle_framework_type import (
    ModelBundleFrameworkType,
)
from launch.api_client.model.model_bundle_packaging_type import (
    ModelBundlePackagingType,
)
from launch.api_client.model.model_endpoint_type import ModelEndpointType
from launch.api_client.model.pytorch_framework import PytorchFramework
from launch.api_client.model.runnable_image_flavor import RunnableImageFlavor
from launch.api_client.model.streaming_enhanced_runnable_image_flavor import (
    StreamingEnhancedRunnableImageFlavor,
)
from launch.api_client.model.tensorflow_framework import TensorflowFramework
from launch.api_client.model.triton_enhanced_runnable_image_flavor import (
    TritonEnhancedRunnableImageFlavor,
)
from launch.api_client.model.update_docker_image_batch_job_v1_request import (
    UpdateDockerImageBatchJobV1Request,
)
from launch.api_client.model.update_model_endpoint_v1_request import (
    UpdateModelEndpointV1Request,
)
from launch.api_client.model.zip_artifact_flavor import ZipArtifactFlavor
from launch.connection import Connection
from launch.constants import (
    BATCH_TASK_INPUT_SIGNED_URL_PATH,
    ENDPOINT_PATH,
    MODEL_BUNDLE_SIGNED_URL_PATH,
    DEFAULT_SCALE_ENDPOINT,
    HOSTED_INFERENCE_PATH,
    LAUNCH_PATH,
)
from launch.docker_image_batch_job_bundle import (
    CreateDockerImageBatchJobBundleResponse,
    DockerImageBatchJobBundleResponse,
    ListDockerImageBatchJobBundleResponse,
)
from launch.find_packages import find_packages_from_imports, get_imports
from launch.hooks import PostInferenceHooks
from launch.make_batch_file import (
    make_batch_input_dict_file,
    make_batch_input_file,
)
from launch.model_bundle import (
    CreateModelBundleV2Response,
    ListModelBundlesV2Response,
    ModelBundle,
    ModelBundleV2Response,
)
from launch.model_endpoint import (
    AsyncEndpoint,
    Endpoint,
    ModelEndpoint,
    StreamingEndpoint,
    SyncEndpoint,
)
from launch.pydantic_schemas import get_model_definitions
from launch.request_validation import validate_task_request

DEFAULT_NETWORK_TIMEOUT_SEC = 120

logger = logging.getLogger(__name__)
logging.basicConfig()

LaunchModel_T = TypeVar("LaunchModel_T")


def _model_bundle_to_name(model_bundle: Union[ModelBundle, str]) -> str:
    if isinstance(model_bundle, ModelBundle):
        return model_bundle.name
    elif isinstance(model_bundle, str):
        return model_bundle
    else:
        raise TypeError("model_bundle should be type ModelBundle or str")


def _model_bundle_to_id(model_bundle: Union[ModelBundle, str]) -> str:
    if isinstance(model_bundle, ModelBundle):
        if model_bundle.id is None:
            raise ValueError(
                "You need to pass in a ModelBundle that has an id, "
                "i.e. one that has already been registered on the server"
            )
        return model_bundle.id
    elif isinstance(model_bundle, str):
        return model_bundle
    else:
        raise TypeError("model_bundle should be type ModelBundle or str")


def _model_endpoint_to_name(model_endpoint: Union[ModelEndpoint, str]) -> str:
    if isinstance(model_endpoint, ModelEndpoint):
        return model_endpoint.name
    elif isinstance(model_endpoint, str):
        return model_endpoint
    else:
        raise TypeError("model_endpoint should be type ModelEndpoint or str")


def _add_app_config_to_bundle_create_payload(payload: Dict[str, Any], app_config: Optional[Union[Dict[str, Any], str]]):
    """
    Edits a request payload (for creating a bundle) to include a (not serialized) app_config if it's
    not None
    """
    if isinstance(app_config, Dict):
        payload["app_config"] = app_config
    elif isinstance(app_config, str):
        with open(app_config, "r") as f:  # pylint: disable=unspecified-encoding
            app_config_dict = yaml.safe_load(f)
            payload["app_config"] = app_config_dict


def _get_model_bundle_framework(
    pytorch_image_tag: Optional[str] = None,
    tensorflow_version: Optional[str] = None,
    custom_base_image_repository: Optional[str] = None,
    custom_base_image_tag: Optional[str] = None,
):
    if pytorch_image_tag is not None:
        return PytorchFramework(
            pytorch_image_tag=pytorch_image_tag,
            framework_type=ModelBundleFrameworkType.PYTORCH,
        )
    elif tensorflow_version is not None:
        return TensorflowFramework(
            tensorflow_version=tensorflow_version,
            framework_type=ModelBundleFrameworkType.TENSORFLOW,
        )
    elif custom_base_image_repository is not None and custom_base_image_tag is not None:
        return CustomFramework(
            image_repository=custom_base_image_repository,
            image_tag=custom_base_image_tag,
            framework_type=ModelBundleFrameworkType.CUSTOM_BASE_IMAGE,
        )
    else:
        raise ValueError(
            "You must specify one of pytorch_image_tag, tensorflow_version, or "
            "custom_base_image_repository and custom_base_image_tag"
        )


def dict_not_none(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


class LaunchClient:
    """Scale Launch Python Client."""

    def __init__(
        self,
        api_key: str,
        endpoint: Optional[str] = None,
        self_hosted: bool = False,
    ):
        """
        Initializes a Scale Launch Client.

        Parameters:
            api_key: Your Scale API key
            endpoint: The Scale Launch Endpoint (this should not need to be changed)
            self_hosted: True iff you are connecting to a self-hosted Scale Launch
        """
        self.endpoint = endpoint or DEFAULT_SCALE_ENDPOINT
        self.connection = Connection(api_key, self.endpoint + HOSTED_INFERENCE_PATH)
        self.self_hosted = self_hosted
        self.upload_bundle_fn: Optional[Callable[[str, str], None]] = None
        self.upload_batch_csv_fn: Optional[Callable[[str, str], None]] = None
        self.bundle_location_fn: Optional[Callable[[], str]] = None
        self.batch_csv_location_fn: Optional[Callable[[], str]] = None
        self.configuration = Configuration(
            host=self.endpoint + LAUNCH_PATH,
            discard_unknown_keys=True,
            username=api_key,
            password="",
        )

    def __repr__(self):
        return f"LaunchClient(connection='{self.connection}')"

    def __eq__(self, other):
        return self.connection == other.connection

    def register_upload_bundle_fn(self, upload_bundle_fn: Callable[[str, str], None]):
        """
        For self-hosted mode only. Registers a function that handles model bundle upload. This
        function is called as

            upload_bundle_fn(serialized_bundle, bundle_url)

        This function should directly write the contents of ``serialized_bundle`` as a
        binary string into ``bundle_url``.

        See ``register_bundle_location_fn`` for more notes on the signature of ``upload_bundle_fn``

        Parameters:
            upload_bundle_fn: Function that takes in a serialized bundle (bytes type),
                and uploads that bundle to an appropriate location. Only needed for self-hosted mode.
        """
        self.upload_bundle_fn = upload_bundle_fn

    def register_upload_batch_csv_fn(self, upload_batch_csv_fn: Callable[[str, str], None]):
        """
        For self-hosted mode only. Registers a function that handles batch text upload. This
        function is called as

            upload_batch_csv_fn(csv_text, csv_url)

        This function should directly write the contents of ``csv_text`` as a text string into
        ``csv_url``.

        Parameters:
            upload_batch_csv_fn: Function that takes in a csv text (string type),
                and uploads that bundle to an appropriate location. Only needed for self-hosted mode.
        """
        self.upload_batch_csv_fn = upload_batch_csv_fn

    def register_bundle_location_fn(self, bundle_location_fn: Callable[[], str]):
        """
        For self-hosted mode only. Registers a function that gives a location for a model bundle.
        Should give different locations each time. This function is called as
        ``bundle_location_fn()``, and should return a ``bundle_url`` that
        ``register_upload_bundle_fn`` can take.

        Strictly, ``bundle_location_fn()`` does not need to return a ``str``. The only
        requirement is that if ``bundle_location_fn`` returns a value of type ``T``,
        then ``upload_bundle_fn()`` takes in an object of type T as its second argument (i.e.
        bundle_url).

        Parameters:
            bundle_location_fn: Function that generates bundle_urls for upload_bundle_fn.
        """
        self.bundle_location_fn = bundle_location_fn

    def register_batch_csv_location_fn(self, batch_csv_location_fn: Callable[[], str]):
        """
        For self-hosted mode only. Registers a function that gives a location for batch CSV
        inputs. Should give different locations each time. This function is called as
        batch_csv_location_fn(), and should return a batch_csv_url that upload_batch_csv_fn can
        take.

        Strictly, batch_csv_location_fn() does not need to return a str. The only requirement is
        that if batch_csv_location_fn returns a value of type T, then upload_batch_csv_fn() takes
        in an object of type T as its second argument (i.e. batch_csv_url).

        Parameters:
            batch_csv_location_fn: Function that generates batch_csv_urls for upload_batch_csv_fn.
        """
        self.batch_csv_location_fn = batch_csv_location_fn

    def _upload_data(self, data: bytes) -> str:
        if self.self_hosted:
            if self.upload_bundle_fn is None:
                raise ValueError("Upload_bundle_fn should be registered")
            if self.bundle_location_fn is None:
                raise ValueError("Need either bundle_location_fn to know where to upload bundles")
            raw_bundle_url = self.bundle_location_fn()  # type: ignore
            self.upload_bundle_fn(data, raw_bundle_url)  # type: ignore
        else:
            model_bundle_url = self.connection.post({}, MODEL_BUNDLE_SIGNED_URL_PATH)
            s3_path = model_bundle_url["signedUrl"]
            raw_bundle_url = f"s3://{model_bundle_url['bucket']}/{model_bundle_url['key']}"
            requests.put(s3_path, data=data)
        return raw_bundle_url

    def _get_bundle_url_from_base_paths(self, base_paths: List[str]) -> str:
        tmpdir = tempfile.mkdtemp()
        try:
            zip_path = os.path.join(tmpdir, "bundle.zip")
            _zip_directories(zip_path, base_paths)
            with open(zip_path, "rb") as zip_f:
                data = zip_f.read()
        finally:
            shutil.rmtree(tmpdir)

        raw_bundle_url = self._upload_data(data)
        return raw_bundle_url

    def _upload_model_bundle(
        self,
        load_model_fn: Callable,
        load_predict_fn: Callable,
    ):
        bundle = dict(load_model_fn=load_model_fn, load_predict_fn=load_predict_fn)
        serialized_bundle = cloudpickle.dumps(bundle)
        bundle_location = self._upload_data(data=serialized_bundle)
        return bundle_location

    def _upload_schemas(self, request_schema: Type[BaseModel], response_schema: Type[BaseModel]) -> str:
        model_definitions = get_model_definitions(
            request_schema=request_schema,
            response_schema=response_schema,
        )
        model_definitions_encoded = json.dumps(model_definitions).encode()
        return self._upload_data(model_definitions_encoded)

    def create_model_bundle_from_callable_v2(
        self,
        *,
        model_bundle_name: str,
        load_predict_fn: Callable[[LaunchModel_T], Callable[[Any], Any]],
        load_model_fn: Callable[[], LaunchModel_T],
        request_schema: Type[BaseModel],
        response_schema: Type[BaseModel],
        requirements: Optional[List[str]] = None,
        pytorch_image_tag: Optional[str] = None,
        tensorflow_version: Optional[str] = None,
        custom_base_image_repository: Optional[str] = None,
        custom_base_image_tag: Optional[str] = None,
        app_config: Optional[Union[Dict[str, Any], str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CreateModelBundleV2Response:
        """
        Uploads and registers a model bundle to Scale Launch.

        Parameters:
            model_bundle_name: Name of the model bundle.

            load_predict_fn: Function that takes in a model and returns a predict function.
                When your model bundle is deployed, this predict function will be called as follows:
                ```
                input = {"input": "some input"} # or whatever your request schema is.

                def load_model_fn():
                    # load model
                    return model

                def load_predict_fn(model, app_config=None):
                    def predict_fn(input):
                        # do pre-processing
                        output = model(input)
                        # do post-processing
                        return output
                    return predict_fn

                predict_fn = load_predict_fn(load_model_fn(), app_config=optional_app_config)
                response = predict_fn(input)
                ```

            load_model_fn: A function that, when run, loads a model.

            request_schema: A pydantic model that represents the request schema for the model
                bundle. This is used to validate the request body for the model bundle's endpoint.

            response_schema: A pydantic model that represents the request schema for the model
                bundle. This is used to validate the response for the model bundle's endpoint.

            requirements: List of pip requirements.

            pytorch_image_tag: The image tag for the PyTorch image that will be used to run the
                bundle. Exactly one of ``pytorch_image_tag``, ``tensorflow_version``, or
                ``custom_base_image_repository`` must be specified.

            tensorflow_version: The version of TensorFlow that will be used to run the bundle.
                If not specified, the default version will be used. Exactly one of
                ``pytorch_image_tag``, ``tensorflow_version``, or ``custom_base_image_repository``
                must be specified.

            custom_base_image_repository: The repository for a custom base image that will be
                used to run the bundle. If not specified, the default base image will be used.
                Exactly one of ``pytorch_image_tag``, ``tensorflow_version``, or
                ``custom_base_image_repository`` must be specified.

            custom_base_image_tag: The tag for a custom base image that will be used to run the
                bundle. Must be specified if ``custom_base_image_repository`` is specified.

            app_config: An optional dictionary of configuration values that will be passed to the
                bundle when it is run. These values can be accessed by the bundle via the
                ``app_config`` global variable.

            metadata: Metadata to record with the bundle.

        Returns:
            An object containing the following keys:

                - ``model_bundle_id``: The ID of the created model bundle.
        """
        nonnull_requirements = requirements or []
        bundle_location = self._upload_model_bundle(load_model_fn, load_predict_fn)
        schema_location = self._upload_schemas(request_schema=request_schema, response_schema=response_schema)
        framework = _get_model_bundle_framework(
            pytorch_image_tag=pytorch_image_tag,
            tensorflow_version=tensorflow_version,
            custom_base_image_repository=custom_base_image_repository,
            custom_base_image_tag=custom_base_image_tag,
        )
        flavor = CloudpickleArtifactFlavor(
            **dict_not_none(
                flavor="cloudpickle_artifact",
                load_predict_fn=inspect.getsource(load_predict_fn),
                load_model_fn=inspect.getsource(load_model_fn),
                framework=framework,
                requirements=nonnull_requirements,
                app_config=app_config,
                location=bundle_location,
            )
        )
        create_model_bundle_request = CreateModelBundleV2Request(
            **dict_not_none(
                name=model_bundle_name,
                schema_location=schema_location,
                flavor=flavor,
                metadata=metadata,
            )
        )
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.create_model_bundle_v2_model_bundles_post(
                body=create_model_bundle_request,
                skip_deserialization=True,
            )
            resp = CreateModelBundleV2Response.parse_raw(response.response.data)

        return resp

    def create_model_bundle_from_dirs_v2(
        self,
        *,
        model_bundle_name: str,
        base_paths: List[str],
        load_predict_fn_module_path: str,
        load_model_fn_module_path: str,
        request_schema: Type[BaseModel],
        response_schema: Type[BaseModel],
        requirements_path: Optional[str] = None,
        pytorch_image_tag: Optional[str] = None,
        tensorflow_version: Optional[str] = None,
        custom_base_image_repository: Optional[str] = None,
        custom_base_image_tag: Optional[str] = None,
        app_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CreateModelBundleV2Response:
        """
        Packages up code from one or more local filesystem folders and uploads them as a bundle
        to Scale Launch. In this mode, a bundle is just local code instead of a serialized object.

        For example, if you have a directory structure like so, and your current working
        directory is ``my_root``:

        ```text
           my_root/
               my_module1/
                   __init__.py
                   ...files and directories
                   my_inference_file.py
               my_module2/
                   __init__.py
                   ...files and directories
        ```

        then calling ``create_model_bundle_from_dirs_v2`` with ``base_paths=["my_module1",
        "my_module2"]`` essentially creates a zip file without the root directory, e.g.:

        ```text
           my_module1/
               __init__.py
               ...files and directories
               my_inference_file.py
           my_module2/
               __init__.py
               ...files and directories
        ```

        and these contents will be unzipped relative to the server side application root. Bear
        these points in mind when referencing Python module paths for this bundle. For instance,
        if ``my_inference_file.py`` has ``def f(...)`` as the desired inference loading function,
        then the `load_predict_fn_module_path` argument should be `my_module1.my_inference_file.f`.

        Parameters:
            model_bundle_name: The name of the model bundle you want to create.

            base_paths: A list of paths to directories that will be zipped up and uploaded
                as a bundle. Each path must be relative to the current working directory.

            load_predict_fn_module_path: The Python module path to the function that will be
                used to load the model for inference. This function should take in a path to a
                model directory, and return a model object. The model object should be pickleable.

            load_model_fn_module_path: The Python module path to the function that will be
                used to load the model for training. This function should take in a path to a
                model directory, and return a model object. The model object should be pickleable.

            request_schema: A Pydantic model that defines the request schema for the bundle.

            response_schema: A Pydantic model that defines the response schema for the bundle.

            requirements_path: Path to a requirements.txt file that will be used to install
                dependencies for the bundle. This file must be relative to the current working
                directory.

            pytorch_image_tag: The image tag for the PyTorch image that will be used to run the
                bundle. Exactly one of ``pytorch_image_tag``, ``tensorflow_version``, or
                ``custom_base_image_repository`` must be specified.

            tensorflow_version: The version of TensorFlow that will be used to run the bundle.
                If not specified, the default version will be used. Exactly one of
                ``pytorch_image_tag``, ``tensorflow_version``, or ``custom_base_image_repository``
                must be specified.

            custom_base_image_repository: The repository for a custom base image that will be
                used to run the bundle. If not specified, the default base image will be used.
                Exactly one of ``pytorch_image_tag``, ``tensorflow_version``, or
                ``custom_base_image_repository`` must be specified.

            custom_base_image_tag: The tag for a custom base image that will be used to run the
                bundle. Must be specified if ``custom_base_image_repository`` is specified.

            app_config: An optional dictionary of configuration values that will be passed to the
                bundle when it is run. These values can be accessed by the bundle via the
                ``app_config`` global variable.

            metadata: Metadata to record with the bundle.

        Returns:
            An object containing the following keys:

                - ``model_bundle_id``: The ID of the created model bundle.
        """
        requirements = []
        if requirements_path is not None:
            with open(requirements_path, "r", encoding="utf-8") as req_f:
                requirements = req_f.read().splitlines()
        bundle_location = self._get_bundle_url_from_base_paths(base_paths)
        schema_location = self._upload_schemas(request_schema=request_schema, response_schema=response_schema)
        framework = _get_model_bundle_framework(
            pytorch_image_tag=pytorch_image_tag,
            tensorflow_version=tensorflow_version,
            custom_base_image_repository=custom_base_image_repository,
            custom_base_image_tag=custom_base_image_tag,
        )
        flavor = ZipArtifactFlavor(
            **dict_not_none(
                flavor="zip_artifact",
                load_predict_fn_module_path=load_predict_fn_module_path,
                load_model_fn_module_path=load_model_fn_module_path,
                framework=framework,
                requirements=requirements,
                app_config=app_config,
                location=bundle_location,
            )
        )
        create_model_bundle_request = CreateModelBundleV2Request(
            **dict_not_none(
                name=model_bundle_name,
                schema_location=schema_location,
                flavor=flavor,
                metadata=metadata,
            )
        )
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.create_model_bundle_v2_model_bundles_post(
                body=create_model_bundle_request,
                skip_deserialization=True,
            )
            resp = CreateModelBundleV2Response.parse_raw(response.response.data)

        return resp

    def create_model_bundle_from_runnable_image_v2(
        self,
        *,
        model_bundle_name: str,
        request_schema: Type[BaseModel],
        response_schema: Type[BaseModel],
        repository: str,
        tag: str,
        command: List[str],
        healthcheck_route: Optional[str] = None,
        predict_route: Optional[str] = None,
        env: Dict[str, str],
        readiness_initial_delay_seconds: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CreateModelBundleV2Response:
        """
        Create a model bundle from a runnable image. The specified ``command`` must start a process
        that will listen for requests on port 5005 using HTTP.

        Inference requests must be served at the `POST /predict` route while the `GET /readyz` route is a healthcheck.

        Parameters:
            model_bundle_name: The name of the model bundle you want to create.

            request_schema: A Pydantic model that defines the request schema for the bundle.

            response_schema: A Pydantic model that defines the response schema for the bundle.

            repository: The name of the Docker repository for the runnable image.

            tag: The tag for the runnable image.

            command: The command that will be used to start the process that listens for requests.

            predict_route: The endpoint route on the runnable image that will be called.

            healthcheck_route: The healthcheck endpoint route on the runnable image.

            env: A dictionary of environment variables that will be passed to the bundle when it
                is run.

            readiness_initial_delay_seconds: The number of seconds to wait for the HTTP server to become ready and
                successfully respond on its healthcheck.

            metadata: Metadata to record with the bundle.

        Returns:
            An object containing the following keys:

                - ``model_bundle_id``: The ID of the created model bundle.
        """
        schema_location = self._upload_schemas(request_schema=request_schema, response_schema=response_schema)
        flavor = RunnableImageFlavor(
            **dict_not_none(
                flavor="runnable_image",
                repository=repository,
                tag=tag,
                command=command,
                healthcheck_route=healthcheck_route,
                predict_route=predict_route,
                env=env,
                protocol="http",
                readiness_initial_delay_seconds=readiness_initial_delay_seconds,
            )
        )
        create_model_bundle_request = CreateModelBundleV2Request(
            **dict_not_none(
                name=model_bundle_name,
                schema_location=schema_location,
                flavor=flavor,
                metadata=metadata,
            )
        )

        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.create_model_bundle_v2_model_bundles_post(
                body=create_model_bundle_request,
                skip_deserialization=True,
            )
            resp = CreateModelBundleV2Response.parse_raw(response.response.data)

        return resp

    def create_model_bundle_from_streaming_enhanced_runnable_image_v2(
        self,
        *,
        model_bundle_name: str,
        request_schema: Type[BaseModel],
        response_schema: Type[BaseModel],
        repository: str,
        tag: str,
        command: Optional[List[str]] = None,
        healthcheck_route: Optional[str] = None,
        predict_route: Optional[str] = None,
        streaming_command: List[str],
        streaming_predict_route: Optional[str] = None,
        env: Dict[str, str],
        readiness_initial_delay_seconds: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CreateModelBundleV2Response:
        """
        Create a model bundle from a runnable image. The specified ``command`` must start a process
        that will listen for requests on port 5005 using HTTP.

        Inference requests must be served at the `POST /predict` route while the `GET /readyz` route is a healthcheck.

        Parameters:
            model_bundle_name: The name of the model bundle you want to create.

            request_schema: A Pydantic model that defines the request schema for the bundle.

            response_schema: A Pydantic model that defines the response schema for the bundle.

            repository: The name of the Docker repository for the runnable image.

            tag: The tag for the runnable image.

            command: The command that will be used to start the process that listens for requests if
                this bundle is used as a SYNC or ASYNC endpoint.

            healthcheck_route: The healthcheck endpoint route on the runnable image.

            predict_route: The endpoint route on the runnable image that will be called if this bundle is used as a SYNC
                or ASYNC endpoint.

            streaming_command: The command that will be used to start the process that listens for
                requests if this bundle is used as a STREAMING endpoint.

            streaming_predict_route: The endpoint route on the runnable image that will be called if this bundle is used
                as a STREAMING endpoint.

            env: A dictionary of environment variables that will be passed to the bundle when it
                is run.

            readiness_initial_delay_seconds: The number of seconds to wait for the HTTP server to become ready and
                successfully respond on its healthcheck.

            metadata: Metadata to record with the bundle.

        Returns:
            An object containing the following keys:

                - ``model_bundle_id``: The ID of the created model bundle.
        """
        schema_location = self._upload_schemas(request_schema=request_schema, response_schema=response_schema)
        flavor = StreamingEnhancedRunnableImageFlavor(
            **dict_not_none(
                flavor="streaming_enhanced_runnable_image",
                repository=repository,
                tag=tag,
                command=command,
                healthcheck_route=healthcheck_route,
                predict_route=predict_route,
                streaming_command=streaming_command,
                streaming_predict_route=streaming_predict_route,
                env=env,
                protocol="http",
                readiness_initial_delay_seconds=readiness_initial_delay_seconds,
            )
        )
        create_model_bundle_request = CreateModelBundleV2Request(
            **dict_not_none(
                name=model_bundle_name,
                schema_location=schema_location,
                flavor=flavor,
                metadata=metadata,
            )
        )

        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.create_model_bundle_v2_model_bundles_post(
                body=create_model_bundle_request,
                skip_deserialization=True,
            )
            resp = CreateModelBundleV2Response.parse_raw(response.response.data)

        return resp

    def create_model_bundle_from_triton_enhanced_runnable_image_v2(
        self,
        *,
        model_bundle_name: str,
        request_schema: Type[BaseModel],
        response_schema: Type[BaseModel],
        repository: str,
        tag: str,
        command: List[str],
        healthcheck_route: Optional[str] = None,
        predict_route: Optional[str] = None,
        env: Dict[str, str],
        readiness_initial_delay_seconds: int,
        triton_model_repository: str,
        triton_model_replicas: Optional[Dict[str, str]] = None,
        triton_num_cpu: float,
        triton_commit_tag: str,
        triton_storage: Optional[str] = None,
        triton_memory: Optional[str] = None,
        triton_readiness_initial_delay_seconds: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CreateModelBundleV2Response:
        """
        Create a model bundle from a runnable image and a tritonserver image.

        Same requirements as :param:`create_model_bundle_from_runnable_image_v2` with additional constraints necessary
        for configuring tritonserver's execution.

        Parameters:
            model_bundle_name: The name of the model bundle you want to create.

            request_schema: A Pydantic model that defines the request schema for the bundle.

            response_schema: A Pydantic model that defines the response schema for the bundle.

            repository: The name of the Docker repository for the runnable image.

            tag: The tag for the runnable image.

            command: The command that will be used to start the process that listens for requests.

            predict_route: The endpoint route on the runnable image that will be called.

            healthcheck_route: The healthcheck endpoint route on the runnable image.

            env: A dictionary of environment variables that will be passed to the bundle when it
                is run.

            readiness_initial_delay_seconds: The number of seconds to wait for the HTTP server to
                become ready and successfully respond on its healthcheck.

            triton_model_repository: The S3 prefix that contains the contents of the model
                repository, formatted according to
                https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md

            triton_model_replicas: If supplied, the name and number of replicas to make for each
                model.

            triton_num_cpu: Number of CPUs, fractional, to allocate to tritonserver.

            triton_commit_tag: The image tag of the specific trionserver version.

            triton_storage: Amount of storage space to allocate for the tritonserver container.

            triton_memory: Amount of memory to allocate for the tritonserver container.

            triton_readiness_initial_delay_seconds: Like readiness_initial_delay_seconds, but for
                tritonserver's own healthcheck.

            metadata: Metadata to record with the bundle.

        Returns:
            An object containing the following keys:

                - ``model_bundle_id``: The ID of the created model bundle.
        """
        schema_location = self._upload_schemas(request_schema=request_schema, response_schema=response_schema)
        flavor = TritonEnhancedRunnableImageFlavor(
            **dict_not_none(
                flavor="triton_enhanced_runnable_image",
                repository=repository,
                tag=tag,
                command=command,
                healthcheck_route=healthcheck_route,
                predict_route=predict_route,
                env=env,
                protocol="http",
                readiness_initial_delay_seconds=readiness_initial_delay_seconds,
                triton_model_repository=triton_model_repository,
                triton_model_replicas=triton_model_replicas,
                triton_num_cpu=triton_num_cpu,
                triton_commit_tag=triton_commit_tag,
                triton_storage=triton_storage,
                triton_memory=triton_memory,
                triton_readiness_initial_delay_seconds=triton_readiness_initial_delay_seconds,
            )
        )
        create_model_bundle_request = CreateModelBundleV2Request(
            **dict_not_none(
                name=model_bundle_name,
                schema_location=schema_location,
                flavor=flavor,
                metadata=metadata,
            )
        )

        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.create_model_bundle_v2_model_bundles_post(
                body=create_model_bundle_request,
                skip_deserialization=True,
            )
            resp = CreateModelBundleV2Response.parse_raw(response.response.data)

        return resp

    def get_model_bundle_v2(self, model_bundle_id: str) -> ModelBundleV2Response:
        """
        Get a model bundle.

        Parameters:
            model_bundle_id: The ID of the model bundle you want to get.

        Returns:
            An object containing the following fields:

                - ``id``: The ID of the model bundle.
                - ``name``: The name of the model bundle.
                - ``flavor``: The flavor of the model bundle. Either `RunnableImage`,
                    `CloudpickleArtifact`, `ZipArtifact`, or `TritonEnhancedRunnableImageFlavor`.
                - ``created_at``: The time the model bundle was created.
                - ``metadata``: A dictionary of metadata associated with the model bundle.
                - ``model_artifact_ids``: A list of IDs of model artifacts associated with the
                    bundle.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            path_params = frozendict({"model_bundle_id": model_bundle_id})
            response = api_instance.get_model_bundle_v2_model_bundles_model_bundle_id_get(  # type: ignore
                path_params=path_params,
                skip_deserialization=True,
            )
            resp = ModelBundleV2Response.parse_raw(response.response.data)

        return resp

    def get_latest_model_bundle_v2(self, model_bundle_name: str) -> ModelBundleV2Response:
        """
        Get the latest version of a model bundle.

        Parameters:
            model_bundle_name: The name of the model bundle you want to get.

        Returns:
            An object containing the following keys:

                - ``id``: The ID of the model bundle.
                - ``name``: The name of the model bundle.
                - ``schema_location``: The location of the schema for the model bundle.
                - ``flavor``: The flavor of the model bundle. Either `RunnableImage`,
                    `CloudpickleArtifact`, `ZipArtifact`, or `TritonEnhancedRunnableImageFlavor`.
                - ``created_at``: The time the model bundle was created.
                - ``metadata``: A dictionary of metadata associated with the model bundle.
                - ``model_artifact_ids``: A list of IDs of model artifacts associated with the
                    bundle.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            query_params = frozendict({"model_name": model_bundle_name})
            response = api_instance.get_latest_model_bundle_v2_model_bundles_latest_get(  # type: ignore
                query_params=query_params,
                skip_deserialization=True,
            )
            resp = ModelBundleV2Response.parse_raw(response.response.data)

        return resp

    def list_model_bundles_v2(self) -> ListModelBundlesV2Response:
        """
        List all model bundles.

        Returns:
            An object containing the following keys:

                - ``model_bundles``: A list of model bundles. Each model bundle is an object.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.list_model_bundles_v2_model_bundles_get(skip_deserialization=True)
            resp = ListModelBundlesV2Response.parse_raw(response.response.data)

        return resp

    def clone_model_bundle_with_changes_v2(
        self,
        original_model_bundle_id: str,
        new_app_config: Optional[Dict[str, Any]] = None,
    ) -> CreateModelBundleV2Response:
        """
        Clone a model bundle with an optional new ``app_config``.

        Parameters:
            original_model_bundle_id: The ID of the model bundle you want to clone.

            new_app_config: A dictionary of new app config values to use for the cloned model.

        Returns:
            An object containing the following keys:

                - ``model_bundle_id``: The ID of the cloned model bundle.
        """
        clone_model_bundle_request = CloneModelBundleV2Request(
            **dict_not_none(
                original_model_bundle_id=original_model_bundle_id,
                new_app_config=new_app_config,
            )
        )
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post(
                body=clone_model_bundle_request,
                skip_deserialization=True,
            )
            resp = CreateModelBundleV2Response.parse_raw(response.response.data)

        return resp

    @deprecated(deprecated_in="1.0.0", details="Use create_model_bundle_from_dirs_v2.")
    def create_model_bundle_from_dirs(
        self,
        *,
        model_bundle_name: str,
        base_paths: List[str],
        requirements_path: str,
        env_params: Dict[str, str],
        load_predict_fn_module_path: str,
        load_model_fn_module_path: str,
        app_config: Optional[Union[Dict[str, Any], str]] = None,
        request_schema: Optional[Type[BaseModel]] = None,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> ModelBundle:
        """
        Warning:
            This method is deprecated. Use
            [``create_model_bundle_from_dirs_v2``](./#launch.client.LaunchClient.create_model_bundle_from_dirs_v2)
            instead.

        Parameters:
            model_bundle_name: The name of the model bundle you want to create. The name
                must be unique across all bundles that you own.

            base_paths: The paths on the local filesystem where the bundle code lives.

            requirements_path: A path on the local filesystem where a ``requirements.txt`` file
                lives.

            env_params: A dictionary that dictates environment information e.g.
                the use of pytorch or tensorflow, which base image tag to use, etc.
                Specifically, the dictionary should contain the following keys:

                - ``framework_type``: either ``tensorflow`` or ``pytorch``.
                - PyTorch fields:
                    - ``pytorch_image_tag``: An image tag for the ``pytorch`` docker base image. The
                        list of tags can be found from https://hub.docker.com/r/pytorch/pytorch/tags

                Example:
                   ```py
                   {
                       "framework_type": "pytorch",
                       "pytorch_image_tag": "1.10.0-cuda11.3-cudnn8-runtime",
                   }
                   ```

            load_predict_fn_module_path: A python module path for a function that, when called
                with the output of load_model_fn_module_path, returns a function that carries out
                inference.

            load_model_fn_module_path: A python module path for a function that returns a model.
                The output feeds into the function located at load_predict_fn_module_path.

            app_config: Either a Dictionary that represents a YAML file contents or a local path
                to a YAML file.

            request_schema: A pydantic model that represents the request schema for the model
                bundle. This is used to validate the request body for the model bundle's endpoint.

            response_schema: A pydantic model that represents the request schema for the model
                bundle. This is used to validate the response for the model bundle's endpoint.
                Note: If request_schema is specified, then response_schema must also be specified.
        """
        with open(requirements_path, "r", encoding="utf-8") as req_f:
            requirements = req_f.read().splitlines()

        raw_bundle_url = self._get_bundle_url_from_base_paths(base_paths)

        schema_location = None
        if bool(request_schema) ^ bool(response_schema):
            raise ValueError("If request_schema is specified, then response_schema must also be specified.")
        if request_schema is not None and response_schema is not None:
            schema_location = self._upload_schemas(request_schema=request_schema, response_schema=response_schema)

        bundle_metadata = {
            "load_predict_fn_module_path": load_predict_fn_module_path,
            "load_model_fn_module_path": load_model_fn_module_path,
        }

        logger.info(
            "create_model_bundle_from_dirs: raw_bundle_url=%s",
            raw_bundle_url,
        )
        payload = dict(
            packaging_type="zip",
            bundle_name=model_bundle_name,
            location=raw_bundle_url,
            bundle_metadata=bundle_metadata,
            requirements=requirements,
            env_params=env_params,
            schema_location=schema_location,
        )
        _add_app_config_to_bundle_create_payload(payload, app_config)

        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            framework = ModelBundleFrameworkType(env_params["framework_type"])
            env_params_copy = env_params.copy()
            env_params_copy["framework_type"] = framework  # type: ignore
            env_params_obj = ModelBundleEnvironmentParams(**env_params_copy)  # type: ignore
            payload = dict_not_none(
                env_params=env_params_obj,
                location=raw_bundle_url,
                name=model_bundle_name,
                requirements=requirements,
                packaging_type=ModelBundlePackagingType("zip"),
                metadata=bundle_metadata,
                app_config=payload.get("app_config"),
                schema_location=schema_location,
            )
            create_model_bundle_request = CreateModelBundleV1Request(**payload)  # type: ignore
            api_instance.create_model_bundle_v1_model_bundles_post(
                body=create_model_bundle_request,
                skip_deserialization=True,
            )
        return ModelBundle(model_bundle_name)

    @deprecated(deprecated_in="1.0.0", details="Use create_model_bundle_from_callable_v2.")
    def create_model_bundle(  # pylint: disable=too-many-statements
        self,
        model_bundle_name: str,
        env_params: Dict[str, str],
        *,
        load_predict_fn: Optional[Callable[[LaunchModel_T], Callable[[Any], Any]]] = None,
        predict_fn_or_cls: Optional[Callable[[Any], Any]] = None,
        requirements: Optional[List[str]] = None,
        model: Optional[LaunchModel_T] = None,
        load_model_fn: Optional[Callable[[], LaunchModel_T]] = None,
        app_config: Optional[Union[Dict[str, Any], str]] = None,
        globals_copy: Optional[Dict[str, Any]] = None,
        request_schema: Optional[Type[BaseModel]] = None,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> ModelBundle:
        """
        Warning:
            This method is deprecated. Use
            [`create_model_bundle_from_callable_v2`](./#create_model_bundle_from_callable_v2) instead.

        Parameters:
            model_bundle_name: The name of the model bundle you want to create. The name
                must be unique across all bundles that you own.

            predict_fn_or_cls: `Function` or a ``Callable`` class that runs end-to-end
                (pre/post processing and model inference) on the call. i.e.
                ``predict_fn_or_cls(REQUEST) -> RESPONSE``.

            model: Typically a trained Neural Network, e.g. a Pytorch module.

                Exactly one of ``model`` and ``load_model_fn`` must be provided.

            load_model_fn: A function that, when run, loads a model. This function is essentially
                a deferred wrapper around the ``model`` argument.

                Exactly one of ``model`` and ``load_model_fn`` must be provided.

            load_predict_fn: Function that, when called with a model, returns a function that
                carries out inference.

                If ``model`` is specified, then this is equivalent
                to:
                    ``load_predict_fn(model, app_config=optional_app_config]) -> predict_fn``

                Otherwise, if ``load_model_fn`` is specified, then this is equivalent to:
                ``load_predict_fn(load_model_fn(), app_config=optional_app_config]) -> predict_fn``

                In both cases, ``predict_fn`` is then the inference function, i.e.:
                    ``predict_fn(REQUEST) -> RESPONSE``


            requirements: A list of python package requirements, where each list element is of
                the form ``<package_name>==<package_version>``, e.g.

                ``["tensorflow==2.3.0", "tensorflow-hub==0.11.0"]``

                If you do not pass in a value for ``requirements``, then you must pass in
                ``globals()`` for the ``globals_copy`` argument.

            app_config: Either a Dictionary that represents a YAML file contents or a local path
                to a YAML file.

            env_params: A dictionary that dictates environment information e.g.
                the use of pytorch or tensorflow, which base image tag to use, etc.
                Specifically, the dictionary should contain the following keys:

                - ``framework_type``: either ``tensorflow`` or ``pytorch``. - PyTorch fields: -
                ``pytorch_image_tag``: An image tag for the ``pytorch`` docker base image. The
                list of tags can be found from https://hub.docker.com/r/pytorch/pytorch/tags. -
                Example:

                    .. code-block:: python

                       {
                           "framework_type": "pytorch",
                           "pytorch_image_tag": "1.10.0-cuda11.3-cudnn8-runtime"
                       }

                - Tensorflow fields:
                    - ``tensorflow_version``: Version of tensorflow, e.g. ``"2.3.0"``.

            globals_copy: Dictionary of the global symbol table. Normally provided by
                ``globals()`` built-in function.

            request_schema: A pydantic model that represents the request schema for the model
                bundle. This is used to validate the request body for the model bundle's endpoint.

            response_schema: A pydantic model that represents the request schema for the model
                bundle. This is used to validate the response for the model bundle's endpoint.
                Note: If request_schema is specified, then response_schema must also be specified.
        """
        # TODO(ivan): remove `disable=too-many-branches` when get rid of `load_*` functions
        # pylint: disable=too-many-branches

        check_args = [
            predict_fn_or_cls is not None,
            load_predict_fn is not None and model is not None,
            load_predict_fn is not None and load_model_fn is not None,
        ]

        if sum(check_args) != 1:
            raise ValueError(
                "A model bundle consists of exactly {predict_fn_or_cls}, {load_predict_fn + "
                "model}, or {load_predict_fn + load_model_fn}. "
            )
        # TODO should we try to catch when people intentionally pass both model and load_model_fn
        #  as None?

        if requirements is None:
            # TODO explore: does globals() actually work as expected? Should we use globals_copy
            #  instead?
            requirements_inferred = find_packages_from_imports(globals())
            requirements = [f"{key}=={value}" for key, value in requirements_inferred.items()]
            logger.info(
                "Using \n%s\n for model bundle %s",
                requirements,
                model_bundle_name,
            )

        # Prepare cloudpickle for external imports
        if globals_copy:
            for module in get_imports(globals_copy):
                if module.__name__ == cloudpickle.__name__:
                    # Avoid recursion
                    # register_pickle_by_value does not work properly with itself
                    continue
                cloudpickle.register_pickle_by_value(module)

        bundle: Union[Callable[[Any], Any], Dict[str, Any], None]  # validate bundle
        bundle_metadata = {}
        # Create bundle
        if predict_fn_or_cls:
            bundle = predict_fn_or_cls
            if inspect.isfunction(predict_fn_or_cls):
                source_code = inspect.getsource(predict_fn_or_cls)
            else:
                source_code = inspect.getsource(predict_fn_or_cls.__class__)
            bundle_metadata["predict_fn_or_cls"] = source_code
        elif model is not None:
            bundle = dict(model=model, load_predict_fn=load_predict_fn)
            bundle_metadata["load_predict_fn"] = inspect.getsource(load_predict_fn)  # type: ignore
        else:
            bundle = dict(load_model_fn=load_model_fn, load_predict_fn=load_predict_fn)
            bundle_metadata["load_predict_fn"] = inspect.getsource(load_predict_fn)  # type: ignore
            bundle_metadata["load_model_fn"] = inspect.getsource(load_model_fn)  # type: ignore

        serialized_bundle = cloudpickle.dumps(bundle)
        raw_bundle_url = self._upload_data(data=serialized_bundle)

        schema_location = None
        if bool(request_schema) ^ bool(response_schema):
            raise ValueError("If request_schema is specified, then response_schema must also be specified.")
        if request_schema is not None and response_schema is not None:
            model_definitions = get_model_definitions(
                request_schema=request_schema,
                response_schema=response_schema,
            )
            model_definitions_encoded = json.dumps(model_definitions).encode()
            schema_location = self._upload_data(model_definitions_encoded)

        payload = dict(
            packaging_type="cloudpickle",
            bundle_name=model_bundle_name,
            location=raw_bundle_url,
            bundle_metadata=bundle_metadata,
            requirements=requirements,
            env_params=env_params,
            schema_location=schema_location,
        )

        _add_app_config_to_bundle_create_payload(payload, app_config)
        framework = ModelBundleFrameworkType(env_params["framework_type"])
        env_params_copy = env_params.copy()
        env_params_copy["framework_type"] = framework  # type: ignore
        env_params_obj = ModelBundleEnvironmentParams(**env_params_copy)  # type: ignore

        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            payload = dict_not_none(
                env_params=env_params_obj,
                location=raw_bundle_url,
                name=model_bundle_name,
                requirements=requirements,
                packaging_type=ModelBundlePackagingType("cloudpickle"),
                metadata=bundle_metadata,
                app_config=app_config,
                schema_location=schema_location,
            )
            create_model_bundle_request = CreateModelBundleV1Request(**payload)  # type: ignore
            api_instance.create_model_bundle_v1_model_bundles_post(
                body=create_model_bundle_request,
                skip_deserialization=True,
            )
        # resp["data"]["name"] should equal model_bundle_name
        # TODO check that a model bundle was created and no name collisions happened
        return ModelBundle(model_bundle_name)

    # pylint: disable=too-many-branches
    def create_model_endpoint(
        self,
        *,
        endpoint_name: str,
        model_bundle: Union[ModelBundle, str],
        cpus: int = 3,
        memory: str = "8Gi",
        storage: Optional[str] = None,
        gpus: int = 0,
        min_workers: int = 1,
        max_workers: int = 1,
        per_worker: int = 10,
        gpu_type: Optional[str] = None,
        endpoint_type: str = "sync",
        high_priority: Optional[bool] = False,
        post_inference_hooks: Optional[List[PostInferenceHooks]] = None,
        default_callback_url: Optional[str] = None,
        default_callback_auth_kind: Optional[Literal["basic", "mtls"]] = None,
        default_callback_auth_username: Optional[str] = None,
        default_callback_auth_password: Optional[str] = None,
        default_callback_auth_cert: Optional[str] = None,
        default_callback_auth_key: Optional[str] = None,
        public_inference: Optional[bool] = None,
        update_if_exists: bool = False,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Endpoint]:
        """
        Creates and registers a model endpoint in Scale Launch. The returned object is an
        instance of type ``Endpoint``, which is a base class of either ``SyncEndpoint`` or
        ``AsyncEndpoint``. This is the object to which you sent inference requests.

        Parameters:
            endpoint_name: The name of the model endpoint you want to create. The name
                must be unique across all endpoints that you own.

            model_bundle: The ``ModelBundle`` that the endpoint should serve.

            cpus: Number of cpus each worker should get, e.g. 1, 2, etc. This must be greater
                than or equal to 1.

            memory: Amount of memory each worker should get, e.g. "4Gi", "512Mi", etc. This must
                be a positive amount of memory.

            storage: Amount of local ephemeral storage each worker should get, e.g. "4Gi",
                "512Mi", etc. This must be a positive amount of storage.

            gpus: Number of gpus each worker should get, e.g. 0, 1, etc.

            min_workers: The minimum number of workers. Must be greater than or equal to 0. This
                should be determined by computing the minimum throughput of your workload and
                dividing it by the throughput of a single worker. This field must be at least ``1``
                for synchronous endpoints.

            max_workers: The maximum number of workers. Must be greater than or equal to 0,
                and as well as greater than or equal to ``min_workers``. This should be determined by
                computing the maximum throughput of your workload and dividing it by the throughput
                of a single worker.

            per_worker: The maximum number of concurrent requests that an individual worker can
                service. Launch automatically scales the number of workers for the endpoint so that
                each worker is processing ``per_worker`` requests, subject to the limits defined by
                ``min_workers`` and ``max_workers``.

                - If the average number of concurrent requests per worker is lower than
                ``per_worker``, then the number of workers will be reduced. - Otherwise,
                if the average number of concurrent requests per worker is higher than
                ``per_worker``, then the number of workers will be increased to meet the elevated
                traffic.

                Here is our recommendation for computing ``per_worker``:

                1. Compute ``min_workers`` and ``max_workers`` per your minimum and maximum
                throughput requirements. 2. Determine a value for the maximum number of
                concurrent requests in the workload. Divide this number by ``max_workers``. Doing
                this ensures that the number of workers will "climb" to ``max_workers``.

            gpu_type: If specifying a non-zero number of gpus, this controls the type of gpu
                requested. Here are the supported values:

                - ``nvidia-tesla-t4``
                - ``nvidia-ampere-a10``

            endpoint_type: Either ``"sync"``, ``"async"``, or ``"streaming"``.

            high_priority: Either ``True`` or ``False``. Enabling this will allow the created
                endpoint to leverage the shared pool of prewarmed nodes for faster spinup time.

            post_inference_hooks: List of hooks to trigger after inference tasks are served.

            default_callback_url: The default callback url to use for async endpoints.
                This can be overridden in the task parameters for each individual task.
                post_inference_hooks must contain "callback" for the callback to be triggered.

            default_callback_auth_kind: The default callback auth kind to use for async endpoints.
                Either "basic" or "mtls". This can be overridden in the task parameters for each
                individual task.

            default_callback_auth_username: The default callback auth username to use. This only
                applies if default_callback_auth_kind is "basic". This can be overridden in the task
                parameters for each individual task.

            default_callback_auth_password: The default callback auth password to use. This only
                applies if default_callback_auth_kind is "basic". This can be overridden in the task
                parameters for each individual task.

            default_callback_auth_cert: The default callback auth cert to use. This only applies
                if default_callback_auth_kind is "mtls". This can be overridden in the task
                parameters for each individual task.

            default_callback_auth_key: The default callback auth key to use. This only applies
                if default_callback_auth_kind is "mtls". This can be overridden in the task
                parameters for each individual task.

            public_inference: If ``True``, this endpoint will be available to all user IDs for
                inference.

            update_if_exists: If ``True``, will attempt to update the endpoint if it exists.
                Otherwise, will unconditionally try to create a new endpoint. Note that endpoint
                names for a given user must be unique, so attempting to call this function with
                ``update_if_exists=False`` for an existing endpoint will raise an error.

            labels: An optional dictionary of key/value pairs to associate with this endpoint.

        Returns:
             A Endpoint object that can be used to make requests to the endpoint.

        """
        existing_endpoint = self.get_model_endpoint(endpoint_name)
        if update_if_exists and existing_endpoint:
            self.edit_model_endpoint(
                model_endpoint=endpoint_name,
                model_bundle=model_bundle,
                cpus=cpus,
                memory=memory,
                storage=storage,
                gpus=gpus,
                min_workers=min_workers,
                max_workers=max_workers,
                per_worker=per_worker,
                gpu_type=gpu_type,
                high_priority=high_priority,
                default_callback_url=default_callback_url,
                default_callback_auth_kind=default_callback_auth_kind,
                default_callback_auth_username=default_callback_auth_username,
                default_callback_auth_password=default_callback_auth_password,
                default_callback_auth_cert=default_callback_auth_cert,
                default_callback_auth_key=default_callback_auth_key,
                public_inference=public_inference,
            )
            return existing_endpoint
        else:
            # Presumably, the user knows that the endpoint doesn't already exist, and so we can
            # defer to the server to reject any duplicate creations.
            logger.info("Creating new endpoint")
            with ApiClient(self.configuration) as api_client:
                api_instance = DefaultApi(api_client)
                if not isinstance(model_bundle, ModelBundle) or model_bundle.id is None:
                    model_bundle = self.get_model_bundle(model_bundle)
                post_inference_hooks_strs = None
                if post_inference_hooks is not None:
                    post_inference_hooks_strs = []
                    for hook in post_inference_hooks:
                        if isinstance(hook, PostInferenceHooks):
                            post_inference_hooks_strs.append(hook.value)
                        else:
                            post_inference_hooks_strs.append(hook)

                if default_callback_auth_kind is not None:
                    default_callback_auth = CallbackAuth(
                        **dict_not_none(
                            kind=default_callback_auth_kind,
                            username=default_callback_auth_username,
                            password=default_callback_auth_password,
                            cert=default_callback_auth_cert,
                            key=default_callback_auth_key,
                        )
                    )
                else:
                    default_callback_auth = None

                payload = dict_not_none(
                    cpus=cpus,
                    endpoint_type=ModelEndpointType(endpoint_type),
                    gpus=gpus,
                    gpu_type=GpuType(gpu_type) if gpu_type is not None else None,
                    labels=labels or {},
                    max_workers=max_workers,
                    memory=memory,
                    metadata={},
                    min_workers=min_workers,
                    model_bundle_id=model_bundle.id,
                    name=endpoint_name,
                    per_worker=per_worker,
                    high_priority=high_priority,
                    post_inference_hooks=post_inference_hooks_strs,
                    default_callback_url=default_callback_url,
                    default_callback_auth=default_callback_auth,
                    storage=storage,
                    public_inference=public_inference,
                )
                create_model_endpoint_request = CreateModelEndpointV1Request(**payload)
                response = api_instance.create_model_endpoint_v1_model_endpoints_post(
                    body=create_model_endpoint_request,
                    skip_deserialization=True,
                )
                resp = json.loads(response.response.data)
            endpoint_creation_task_id = resp.get("endpoint_creation_task_id", None)  # TODO probably throw on None
            logger.info("Endpoint creation task id is %s", endpoint_creation_task_id)
            model_endpoint = ModelEndpoint(name=endpoint_name, bundle_name=model_bundle.name)
            if endpoint_type == "async":
                return AsyncEndpoint(model_endpoint=model_endpoint, client=self)
            elif endpoint_type == "sync":
                return SyncEndpoint(model_endpoint=model_endpoint, client=self)
            elif endpoint_type == "streaming":
                return StreamingEndpoint(model_endpoint=model_endpoint, client=self)
            else:
                raise ValueError("Endpoint should be one of the types 'sync', 'async', or 'streaming'")

    def edit_model_endpoint(
        self,
        *,
        model_endpoint: Union[ModelEndpoint, str],
        model_bundle: Optional[Union[ModelBundle, str]] = None,
        cpus: Optional[float] = None,
        memory: Optional[str] = None,
        storage: Optional[str] = None,
        gpus: Optional[int] = None,
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        per_worker: Optional[int] = None,
        gpu_type: Optional[str] = None,
        high_priority: Optional[bool] = None,
        post_inference_hooks: Optional[List[PostInferenceHooks]] = None,
        default_callback_url: Optional[str] = None,
        default_callback_auth_kind: Optional[Literal["basic", "mtls"]] = None,
        default_callback_auth_username: Optional[str] = None,
        default_callback_auth_password: Optional[str] = None,
        default_callback_auth_cert: Optional[str] = None,
        default_callback_auth_key: Optional[str] = None,
        public_inference: Optional[bool] = None,
    ) -> None:
        """
        Edits an existing model endpoint. Here are the fields that **cannot** be edited on an
        existing endpoint:

        - The endpoint's name. - The endpoint's type (i.e. you cannot go from a ``SyncEnpdoint``
        to an ``AsyncEndpoint`` or vice versa.

        Parameters:
            model_endpoint: The model endpoint (or its name) you want to edit. The name
                must be unique across all endpoints that you own.

            model_bundle: The ``ModelBundle`` that the endpoint should serve.

            cpus: Number of cpus each worker should get, e.g. 1, 2, etc. This must be greater
                than or equal to 1.

            memory: Amount of memory each worker should get, e.g. "4Gi", "512Mi", etc. This must
                be a positive amount of memory.

            storage: Amount of local ephemeral storage each worker should get, e.g. "4Gi",
                "512Mi", etc. This must be a positive amount of storage.

            gpus: Number of gpus each worker should get, e.g. 0, 1, etc.

            min_workers: The minimum number of workers. Must be greater than or equal to 0.

            max_workers: The maximum number of workers. Must be greater than or equal to 0,
                and as well as greater than or equal to ``min_workers``.

            per_worker: The maximum number of concurrent requests that an individual worker can
                service. Launch automatically scales the number of workers for the endpoint so that
                each worker is processing ``per_worker`` requests:

                - If the average number of concurrent requests per worker is lower than
                ``per_worker``, then the number of workers will be reduced. - Otherwise,
                if the average number of concurrent requests per worker is higher than
                ``per_worker``, then the number of workers will be increased to meet the elevated
                traffic.

            gpu_type: If specifying a non-zero number of gpus, this controls the type of gpu
                requested. Here are the supported values:

                - ``nvidia-tesla-t4``
                - ``nvidia-ampere-a10``

            high_priority: Either ``True`` or ``False``. Enabling this will allow the created
                endpoint to leverage the shared pool of prewarmed nodes for faster spinup time.

            post_inference_hooks: List of hooks to trigger after inference tasks are served.

            default_callback_url: The default callback url to use for async endpoints.
                This can be overridden in the task parameters for each individual task.
                post_inference_hooks must contain "callback" for the callback to be triggered.

            default_callback_auth_kind: The default callback auth kind to use for async endpoints.
                Either "basic" or "mtls". This can be overridden in the task parameters for each
                individual task.

            default_callback_auth_username: The default callback auth username to use. This only
                applies if default_callback_auth_kind is "basic". This can be overridden in the task
                parameters for each individual task.

            default_callback_auth_password: The default callback auth password to use. This only
                applies if default_callback_auth_kind is "basic". This can be overridden in the task
                parameters for each individual task.

            default_callback_auth_cert: The default callback auth cert to use. This only applies
                if default_callback_auth_kind is "mtls". This can be overridden in the task
                parameters for each individual task.

            default_callback_auth_key: The default callback auth key to use. This only applies
                if default_callback_auth_kind is "mtls". This can be overridden in the task
                parameters for each individual task.

            public_inference: If ``True``, this endpoint will be available to all user IDs for
                inference.
        """
        logger.info("Editing existing endpoint")
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)

            if model_bundle is None:
                model_bundle_id = None
            elif isinstance(model_bundle, ModelBundle) and model_bundle.id is not None:
                model_bundle_id = model_bundle.id
            else:
                model_bundle = self.get_model_bundle(model_bundle)
                model_bundle_id = model_bundle.id

            if model_endpoint is None:
                model_endpoint_id = None
            elif isinstance(model_endpoint, ModelEndpoint) and model_endpoint.id is not None:
                model_endpoint_id = model_endpoint.id
            else:
                endpoint_name = _model_endpoint_to_name(model_endpoint)
                model_endpoint_full = self.get_model_endpoint(endpoint_name)
                model_endpoint_id = model_endpoint_full.model_endpoint.id  # type: ignore

            post_inference_hooks_strs = None
            if post_inference_hooks is not None:
                post_inference_hooks_strs = []
                for hook in post_inference_hooks:
                    if isinstance(hook, PostInferenceHooks):
                        post_inference_hooks_strs.append(hook.value)
                    else:
                        post_inference_hooks_strs.append(hook)

            if default_callback_auth_kind is not None:
                default_callback_auth = CallbackAuth(
                    **dict_not_none(
                        kind=default_callback_auth_kind,
                        username=default_callback_auth_username,
                        password=default_callback_auth_password,
                        cert=default_callback_auth_cert,
                        key=default_callback_auth_key,
                    )
                )
            else:
                default_callback_auth = None

            payload = dict_not_none(
                cpus=cpus,
                gpus=gpus,
                gpu_type=GpuType(gpu_type) if gpu_type is not None else None,
                max_workers=max_workers,
                memory=memory,
                min_workers=min_workers,
                model_bundle_id=model_bundle_id,
                per_worker=per_worker,
                high_priority=high_priority,
                post_inference_hooks=post_inference_hooks_strs,
                default_callback_url=default_callback_url,
                default_callback_auth=default_callback_auth,
                storage=storage,
                public_inference=public_inference,
            )
            update_model_endpoint_request = UpdateModelEndpointV1Request(**payload)
            path_params = frozendict({"model_endpoint_id": model_endpoint_id})
            response = api_instance.update_model_endpoint_v1_model_endpoints_model_endpoint_id_put(  # type: ignore
                body=update_model_endpoint_request,
                path_params=path_params,  # type: ignore
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
        endpoint_creation_task_id = resp.get("endpoint_creation_task_id", None)  # Returned from server as "creation"
        logger.info("Endpoint edit task id is %s", endpoint_creation_task_id)

    def get_model_endpoint(self, endpoint_name: str) -> Optional[Union[AsyncEndpoint, SyncEndpoint]]:
        """
        Gets a model endpoint associated with a name.

        Parameters:
            endpoint_name: The name of the endpoint to retrieve.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            query_params = frozendict({"name": endpoint_name})
            response = api_instance.list_model_endpoints_v1_model_endpoints_get(  # type: ignore
                query_params=query_params,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
            if len(resp["model_endpoints"]) == 0:
                return None
            resp = resp["model_endpoints"][0]

        if resp["endpoint_type"] == "async":
            return AsyncEndpoint(ModelEndpoint.from_dict(resp), client=self)  # type: ignore
        elif resp["endpoint_type"] == "sync":
            return SyncEndpoint(ModelEndpoint.from_dict(resp), client=self)  # type: ignore
        elif resp["endpoint_type"] == "streaming":
            return StreamingEndpoint(ModelEndpoint.from_dict(resp), client=self)  # type: ignore
        else:
            raise ValueError("Endpoint should be one of the types 'sync', 'async', or 'streaming'")

    def list_model_bundles(self) -> List[ModelBundle]:
        """
        Returns a list of model bundles that the user owns.

        Returns:
            A list of ModelBundle objects
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.list_model_bundles_v1_model_bundles_get(skip_deserialization=True)
            resp = json.loads(response.response.data)
        model_bundles = [ModelBundle.from_dict(item) for item in resp["model_bundles"]]  # type: ignore
        return model_bundles

    def get_model_bundle(self, model_bundle: Union[ModelBundle, str]) -> ModelBundle:
        """
        Returns a model bundle specified by ``bundle_name`` that the user owns.

        Parameters:
            model_bundle: The bundle or its name.

        Returns:
            A ``ModelBundle`` object
        """
        bundle_name = _model_bundle_to_name(model_bundle)
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            query_params = frozendict({"model_name": bundle_name})
            response = api_instance.get_latest_model_bundle_v1_model_bundles_latest_get(  # type: ignore
                query_params=query_params,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
        return ModelBundle.from_dict(resp)  # type: ignore

    @deprecated(deprecated_in="1.0.0", details="Use create_model_bundle_from_callable_v2.")
    def clone_model_bundle_with_changes(
        self,
        model_bundle: Union[ModelBundle, str],
        app_config: Optional[Dict] = None,
    ) -> ModelBundle:
        """
        Warning:
            This method is deprecated. Use
            [`clone_model_bundle_with_changes_v2`](./#clone_model_bundle_with_changes_v2) instead.

        Parameters:
            model_bundle: The existing bundle or its ID.
            app_config: The new bundle's app config, if not passed in, the new
                bundle's ``app_config`` will be set to ``None``

        Returns:
            A ``ModelBundle`` object
        """

        bundle_id = _model_bundle_to_id(model_bundle)
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            payload = dict_not_none(
                original_model_bundle_id=bundle_id,
                new_app_config=app_config,
            )
            clone_model_bundle_request = CloneModelBundleV1Request(**payload)
            response = (
                api_instance.clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post(  # noqa: E501
                    body=clone_model_bundle_request,
                    skip_deserialization=True,
                )
            )
        return json.loads(response.response.data)

    def list_model_endpoints(self) -> List[Endpoint]:
        """
        Lists all model endpoints that the user owns.

        Returns:
            A list of ``ModelEndpoint`` objects.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.list_model_endpoints_v1_model_endpoints_get(skip_deserialization=True)
            resp = json.loads(response.response.data)
        async_endpoints: List[Endpoint] = [
            AsyncEndpoint(
                model_endpoint=ModelEndpoint.from_dict(endpoint),  # type: ignore
                client=self,
            )
            for endpoint in resp["model_endpoints"]
            if endpoint["endpoint_type"] == "async"
        ]
        sync_endpoints: List[Endpoint] = [
            SyncEndpoint(
                model_endpoint=ModelEndpoint.from_dict(endpoint),  # type: ignore
                client=self,
            )
            for endpoint in resp["model_endpoints"]
            if endpoint["endpoint_type"] == "sync"
        ]
        streaming_endpoints: List[Endpoint] = [
            StreamingEndpoint(
                model_endpoint=ModelEndpoint.from_dict(endpoint),  # type: ignore
                client=self,
            )
            for endpoint in resp["model_endpoints"]
            if endpoint["endpoint_type"] == "streaming"
        ]
        return async_endpoints + sync_endpoints + streaming_endpoints

    def delete_model_endpoint(self, model_endpoint: Union[ModelEndpoint, str]):
        """
        Deletes a model endpoint.

        Parameters:
            model_endpoint: A ``ModelEndpoint`` object.
        """
        endpoint_name = _model_endpoint_to_name(model_endpoint)
        endpoint = self.get_model_endpoint(endpoint_name)
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            model_endpoint_id = endpoint.model_endpoint.id  # type: ignore
            path_params = frozendict({"model_endpoint_id": model_endpoint_id})
            response = api_instance.delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete(  # type: ignore
                path_params=path_params,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
        return resp["deleted"]

    def read_endpoint_creation_logs(self, model_endpoint: Union[ModelEndpoint, str]):
        """
        Retrieves the logs for the creation of the endpoint.

        Parameters:
            model_endpoint: The endpoint or its name.
        """
        endpoint_name = _model_endpoint_to_name(model_endpoint)
        route = f"{ENDPOINT_PATH}/creation_logs/{endpoint_name}"
        resp = self.connection.get(route)
        return resp["content"]

    def _streaming_request(
        self,
        endpoint_name: str,
        url: Optional[str] = None,
        args: Optional[Dict] = None,
        return_pickled: bool = False,
    ) -> requests.Response:
        """
        Not recommended for use, instead use functions provided by StreamingEndpoint. Makes a
        request to the Sync Model Endpoint at endpoint_id, and blocks until request completion or
        timeout. Endpoint at endpoint_id must be a SyncEndpoint, otherwise this request will fail.

        Parameters:
            endpoint_name: The name of the endpoint to make the request to

            url: A url that points to a file containing model input. Must be accessible by Scale
            Launch, hence it needs to either be public or a signedURL. **Note**: the contents of
            the file located at ``url`` are opened as a sequence of ``bytes`` and passed to the
            predict function. If you instead want to pass the url itself as an input to the
            predict function, see ``args``.

            args: A dictionary of arguments to the ``predict`` function defined in your model
            bundle. Must be json-serializable, i.e. composed of ``str``, ``int``, ``float``,
            etc. If your ``predict`` function has signature ``predict(foo, bar)``, then args
            should be a dictionary with keys ``foo`` and ``bar``. Exactly one of ``url`` and
            ``args`` must be specified.

            return_pickled: Whether the python object returned is pickled, or directly written to
            the file returned.

        Returns:
            A requests.Response object.
        """
        validate_task_request(url=url, args=args)
        endpoint = self.get_model_endpoint(endpoint_name)
        endpoint_id = endpoint.model_endpoint.id  # type: ignore
        payload = dict_not_none(return_pickled=return_pickled, url=url, args=args)
        response = requests.post(
            url=f"{self.configuration.host}/v1/streaming-tasks?model_endpoint_id={endpoint_id}",
            json=payload,
            auth=(self.configuration.username, self.configuration.password),
        )
        return response

    def _sync_request(
        self,
        endpoint_name: str,
        url: Optional[str] = None,
        args: Optional[Dict] = None,
        return_pickled: bool = False,
    ) -> Dict[str, Any]:
        """
        Not recommended for use, instead use functions provided by SyncEndpoint Makes a request
        to the Sync Model Endpoint at endpoint_id, and blocks until request completion or
        timeout. Endpoint at endpoint_id must be a SyncEndpoint, otherwise this request will fail.

        Parameters:
            endpoint_name: The name of the endpoint to make the request to

            url: A url that points to a file containing model input. Must be accessible by Scale
            Launch, hence it needs to either be public or a signedURL. **Note**: the contents of
            the file located at ``url`` are opened as a sequence of ``bytes`` and passed to the
            predict function. If you instead want to pass the url itself as an input to the
            predict function, see ``args``.

            args: A dictionary of arguments to the ``predict`` function defined in your model
            bundle. Must be json-serializable, i.e. composed of ``str``, ``int``, ``float``,
            etc. If your ``predict`` function has signature ``predict(foo, bar)``, then args
            should be a dictionary with keys ``foo`` and ``bar``. Exactly one of ``url`` and
            ``args`` must be specified.

            return_pickled: Whether the python object returned is pickled, or directly written to
            the file returned.

        Returns:
            A dictionary with key either ``"result_url"`` or ``"result"``, depending on the value
            of ``return_pickled``. If ``return_pickled`` is true, the key will be ``"result_url"``,
            and the value is a signedUrl that contains a cloudpickled Python object,
            the result of running inference on the model input.
            Example output:
                ``https://foo.s3.us-west-2.amazonaws.com/bar/baz/qux?xyzzy``

            Otherwise, if ``return_pickled`` is false, the key will be ``"result"``,
            and the value is the output of the endpoint's ``predict`` function, serialized as json.
        """
        validate_task_request(url=url, args=args)
        endpoint = self.get_model_endpoint(endpoint_name)
        endpoint_id = endpoint.model_endpoint.id  # type: ignore
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            payload = dict_not_none(return_pickled=return_pickled, url=url, args=args)
            request = EndpointPredictV1Request(**payload)
            query_params = frozendict({"model_endpoint_id": endpoint_id})
            response = api_instance.create_sync_inference_task_v1_sync_tasks_post(  # type: ignore
                body=request,
                query_params=query_params,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
        return resp

    def _async_request(
        self,
        endpoint_name: str,
        *,
        url: Optional[str] = None,
        args: Optional[Dict] = None,
        callback_url: Optional[str] = None,
        callback_auth_kind: Optional[Literal["basic", "mtls"]] = None,
        callback_auth_username: Optional[str] = None,
        callback_auth_password: Optional[str] = None,
        callback_auth_cert: Optional[str] = None,
        callback_auth_key: Optional[str] = None,
        return_pickled: bool = False,
    ) -> str:
        """
        Makes a request to the Async Model Endpoint at endpoint_id, and immediately returns a key
        that can be used to retrieve the result of inference at a later time.

        Parameters:
            endpoint_name: The name of the endpoint to make the request to

            url: A url that points to a file containing model input. Must be accessible by Scale
            Launch, hence it needs to either be public or a signedURL. **Note**: the contents of
            the file located at ``url`` are opened as a sequence of ``bytes`` and passed to the
            predict function. If you instead want to pass the url itself as an input to the
            predict function, see ``args``.

            args: A dictionary of arguments to the ModelBundle's predict function. Must be
            json-serializable, i.e. composed of ``str``, ``int``, ``float``, etc. If your predict
            function has signature ``predict(foo, bar)``, then args should be a dictionary with
            keys ``"foo"`` and ``"bar"``.

                Exactly one of ``url`` and ``args`` must be specified.

            callback_url: The callback url to use for this task. If None, then the
                default_callback_url of the endpoint is used. The endpoint must specify
                "callback" as a post-inference hook for the callback to be triggered.

            callback_auth_kind: The default callback auth kind to use for async endpoints.
                Either "basic" or "mtls". This can be overridden in the task parameters for each
                individual task.

            callback_auth_username: The default callback auth username to use. This only
                applies if callback_auth_kind is "basic". This can be overridden in the task
                parameters for each individual task.

            callback_auth_password: The default callback auth password to use. This only
                applies if callback_auth_kind is "basic". This can be overridden in the task
                parameters for each individual task.

            callback_auth_cert: The default callback auth cert to use. This only applies
                if callback_auth_kind is "mtls". This can be overridden in the task
                parameters for each individual task.

            callback_auth_key: The default callback auth key to use. This only applies
                if callback_auth_kind is "mtls". This can be overridden in the task
                parameters for each individual task.

            return_pickled: Whether the python object returned is pickled, or directly written to
            the file returned.

        Returns:
            An id/key that can be used to fetch inference results at a later time.
            Example output:
                `abcabcab-cabc-abca-0123456789ab`
        """
        validate_task_request(url=url, args=args)
        endpoint = self.get_model_endpoint(endpoint_name)
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            if callback_auth_kind is not None:
                callback_auth = CallbackAuth(
                    **dict_not_none(
                        kind=callback_auth_kind,
                        username=callback_auth_username,
                        password=callback_auth_password,
                        cert=callback_auth_cert,
                        key=callback_auth_key,
                    )
                )
            else:
                callback_auth = None

            payload = dict_not_none(
                return_pickled=return_pickled,
                url=url,
                args=args,
                callback_url=callback_url,
                callback_auth=callback_auth,
            )
            request = EndpointPredictV1Request(**payload)
            model_endpoint_id = endpoint.model_endpoint.id  # type: ignore
            query_params = frozendict({"model_endpoint_id": model_endpoint_id})
            response = api_instance.create_async_inference_task_v1_async_tasks_post(  # type: ignore
                body=request,
                query_params=query_params,  # type: ignore
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
        return resp

    def _get_async_endpoint_response(self, endpoint_name: str, async_task_id: str) -> Dict[str, Any]:
        """
        Not recommended to use this, instead we recommend to use functions provided by
        AsyncEndpoint. Gets inference results from a previously created task.

        Parameters:
            endpoint_name: The name of the endpoint the request was made to.
            async_task_id: The id/key returned from a previous invocation of async_request.

        Returns: A dictionary that contains task status and optionally a result url or result if
        the task has completed. Result url or result will be returned if the task has succeeded.
        Will return a result url iff ``return_pickled`` was set to ``True`` on task creation.

            The dictionary's keys are as follows:

            - ``status``: ``'PENDING'`` or ``'SUCCESS'`` or ``'FAILURE'`` - ``result_url``: a url
            pointing to inference results. This url is accessible for 12 hours after the request
            has been made. - ``result``: the value returned by the endpoint's `predict` function,
            serialized as json

            Example output:

            .. code-block:: json

                {
                    'status': 'SUCCESS',
                    'result_url': 'https://foo.s3.us-west-2.amazonaws.com/bar/baz/qux?xyzzy'
                }

        """
        # TODO: do we want to read the results from here as well? i.e. translate result_url into
        #  a python object
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            path_params = frozendict({"task_id": async_task_id})
            response = api_instance.get_async_inference_task_v1_async_tasks_task_id_get(  # type: ignore
                path_params=path_params,  # type: ignore
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
        return resp

    def batch_async_request(
        self,
        *,
        model_bundle: Union[ModelBundle, str],
        urls: Optional[List[str]] = None,
        inputs: Optional[List[Dict[str, Any]]] = None,
        batch_url_file_location: Optional[str] = None,
        serialization_format: str = "JSON",
        labels: Optional[Dict[str, str]] = None,
        cpus: Optional[int] = None,
        memory: Optional[str] = None,
        gpus: Optional[int] = None,
        gpu_type: Optional[str] = None,
        storage: Optional[str] = None,
        max_workers: Optional[int] = None,
        per_worker: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Sends a batch inference request using a given bundle. Returns a key that can be used to
        retrieve the results of inference at a later time.

        Must have exactly one of urls or inputs passed in.

        Parameters:
            model_bundle: The bundle or the name of a the bundle to use for inference.

            urls: A list of urls, each pointing to a file containing model input. Must be
                accessible by Scale Launch, hence urls need to either be public or signedURLs.

            inputs: A list of model inputs, if exists, we will upload the inputs and pass it in
                to Launch.

            batch_url_file_location: In self-hosted mode, the input to the batch job will be
                uploaded to this location if provided. Otherwise, one will be determined from
                bundle_location_fn()

            serialization_format: Serialization format of output, either 'PICKLE' or 'JSON'.
                'pickle' corresponds to pickling results + returning

            labels: An optional dictionary of key/value pairs to associate with this endpoint.

            cpus: Number of cpus each worker should get, e.g. 1, 2, etc. This must be greater than
                or equal to 1.

            memory: Amount of memory each worker should get, e.g. "4Gi", "512Mi", etc. This must be
                a positive amount of memory.

            storage: Amount of local ephemeral storage each worker should get, e.g. "4Gi", "512Mi",
                etc. This must be a positive amount of storage.

            gpus: Number of gpus each worker should get, e.g. 0, 1, etc.

            max_workers: The maximum number of workers. Must be greater than or equal to 0, and as
                well as greater than or equal to ``min_workers``.

            per_worker: The maximum number of concurrent requests that an individual worker can
                service. Launch automatically scales the number of workers for the endpoint so that
                each worker is processing ``per_worker`` requests:

                - If the average number of concurrent requests per worker is lower than
                  ``per_worker``, then the number of workers will be reduced.
                - Otherwise, if the average number of concurrent requests per worker is higher
                  than ``per_worker``, then the number of workers will be increased to meet the
                  elevated traffic.

            gpu_type: If specifying a non-zero number of gpus, this controls the type of gpu
                requested. Here are the supported values:

                - ``nvidia-tesla-t4``
                - ``nvidia-ampere-a10``

            timeout_seconds: The maximum amount of time (in seconds) that the batch job can take.
                If not specified, the server defaults to 12 hours. This includes the time required
                to build the endpoint and the total time required for all the individual tasks.

        Returns:
            A dictionary that contains `job_id` as a key, and the ID as the value.
        """

        if not bool(inputs) ^ bool(urls):
            raise ValueError("Exactly one of inputs and urls is required for batch tasks")

        f = StringIO()
        if urls:
            make_batch_input_file(urls, f)
        elif inputs:
            make_batch_input_dict_file(inputs, f)
        f.seek(0)

        if self.self_hosted:
            # TODO make this not use bundle_location_fn()
            location_fn = self.batch_csv_location_fn or self.bundle_location_fn
            if location_fn is None and batch_url_file_location is None:
                raise ValueError("Must register batch_csv_location_fn if csv file location not passed in")
            file_location = batch_url_file_location or location_fn()  # type: ignore
            self.upload_batch_csv_fn(f.getvalue(), file_location)  # type: ignore
        else:
            model_bundle_s3_url = self.connection.post({}, BATCH_TASK_INPUT_SIGNED_URL_PATH)
            s3_path = model_bundle_s3_url["signedUrl"]
            requests.put(s3_path, data=f.getvalue())
            file_location = f"s3://{model_bundle_s3_url['bucket']}/{model_bundle_s3_url['key']}"

        logger.info("Writing batch task csv to %s", file_location)

        if not isinstance(model_bundle, ModelBundle) or model_bundle.id is None:
            model_bundle = self.get_model_bundle(model_bundle)

        resource_requests = dict_not_none(
            cpus=cpus,
            memory=memory,
            gpus=gpus,
            gpu_type=gpu_type,
            storage=storage,
            max_workers=max_workers,
            per_worker=per_worker,
        )
        payload = dict_not_none(
            model_bundle_id=model_bundle.id,
            input_path=file_location,
            serialization_format=serialization_format,
            labels=labels,
            resource_requests=resource_requests,
            timeout_seconds=timeout_seconds,
        )
        request = CreateBatchJobV1Request(**payload)
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.create_batch_job_v1_batch_jobs_post(  # type: ignore
                body=request,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
        return resp

    def get_batch_async_response(self, batch_job_id: str) -> Dict[str, Any]:
        """
        Gets inference results from a previously created batch job.

        Parameters:
            batch_job_id: An id representing the batch task job. This id is the in the response from
                calling ``batch_async_request``.

        Returns:
            A dictionary that contains the following fields:

            - ``status``: The status of the job.
            - ``result``: The url where the result is stored.
            - ``duration``: A string representation of how long the job took to finish
                    or how long it has been running, for a job current in progress.
            - ``num_tasks_pending``: The number of tasks that are still pending.
            - ``num_tasks_completed``: The number of tasks that have completed.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            path_params = frozendict({"batch_job_id": batch_job_id})
            response = api_instance.get_batch_job_v1_batch_jobs_batch_job_id_get(  # type: ignore
                path_params=path_params,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
        return resp

    def create_docker_image_batch_job_bundle(
        self,
        *,
        name: str,
        image_repository: str,
        image_tag: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        mount_location: Optional[str] = None,
        cpus: Optional[int] = None,
        memory: Optional[str] = None,
        gpus: Optional[int] = None,
        gpu_type: Optional[str] = None,
        storage: Optional[str] = None,
    ) -> CreateDockerImageBatchJobBundleResponse:
        """
        For self hosted mode only.

        Creates a Docker Image Batch Job Bundle.

        Parameters:
            name:
                A user-defined name for the bundle. Does not need to be unique.
            image_repository:
                The (short) repository of your image. For example, if your image is located at
                123456789012.dkr.ecr.us-west-2.amazonaws.com/repo:tag, and your version of Launch
                is configured to look at 123456789012.dkr.ecr.us-west-2.amazonaws.com for Docker Images,
                you would pass the value `repo` for the `image_repository` parameter.
            image_tag:
                The tag of your image inside of the repo. In the example above, you would pass
                the value `tag` for the `image_tag` parameter.
            command:
                The command to run inside the docker image.
            env:
                A dictionary of environment variables to inject into your docker image.
            mount_location:
                A location in the filesystem where you would like a json-formatted file, controllable
                on runtime, to be mounted. This allows behavior to be specified on runtime.
                (Specifically, the contents of this file can be read via `json.load()` inside of the
                user-defined code.)
            cpus:
                Optional default value for the number of cpus to give the job.
            memory:
                Optional default value for the amount of memory to give the job.
            gpus:
                Optional default value for the number of gpus to give the job.
            gpu_type:
                Optional default value for the type of gpu to give the job.
            storage:
                Optional default value for the amount of disk to give the job.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            # Raw dictionary since it's not the entire request and we can get away with this
            # Also having it be a CreateDockerImageBatchJobResourceRequest runs into problems
            # if no values are specified
            resource_requests = dict_not_none(
                cpus=cpus,
                memory=memory,
                gpus=gpus,
                gpu_type=gpu_type,
                storage=storage,
            )
            create_docker_image_batch_job_bundle_request = CreateDockerImageBatchJobBundleV1Request(
                **dict_not_none(
                    name=name,
                    image_repository=image_repository,
                    image_tag=image_tag,
                    command=command,
                    env=env,
                    mount_location=mount_location,
                    resource_requests=resource_requests,
                )
            )
            response = api_instance.create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post(
                body=create_docker_image_batch_job_bundle_request, skip_deserialization=True
            )
            resp = CreateDockerImageBatchJobBundleResponse.parse_raw(response.response.data)
        return resp

    def get_docker_image_batch_job_bundle(
        self, docker_image_batch_job_bundle_id: str
    ) -> DockerImageBatchJobBundleResponse:
        """
        For self hosted mode only. Gets information for a single batch job bundle with a given id.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            path_params = frozendict({"docker_image_batch_job_bundle_id": docker_image_batch_job_bundle_id})
            response = api_instance.get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get(  # type: ignore  # noqa: E501
                path_params=path_params,
                skip_deserialization=True,
            )
            resp = DockerImageBatchJobBundleResponse.parse_raw(response.response.data)

        return resp

    def get_latest_docker_image_batch_job_bundle(self, bundle_name: str) -> DockerImageBatchJobBundleResponse:
        """
        For self hosted mode only. Gets information for the latest batch job bundle with a given name.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            query_params = frozendict({"bundle_name": bundle_name})
            response = api_instance.get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get(  # type: ignore  # noqa: E501
                query_params=query_params,
                skip_deserialization=True,
            )
            resp = DockerImageBatchJobBundleResponse.parse_raw(response.response.data)

        return resp

    def list_docker_image_batch_job_bundles(
        self, bundle_name: Optional[str] = None, order_by: Optional[Literal["newest", "oldest"]] = None
    ) -> ListDockerImageBatchJobBundleResponse:
        """
        For self hosted mode only. Gets information for multiple bundles.

        Parameters:
            bundle_name: The name of the bundles to retrieve. If not specified, this will retrieve all
            bundles.
            order_by: Either "newest", "oldest", or not specified. Specify to sort by newest/oldest.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            query_params = frozendict(dict_not_none(bundle_name=bundle_name, order_by=order_by))
            response = api_instance.list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get(  # type: ignore  # noqa: E501
                query_params=query_params,
                skip_deserialization=True,
            )
            resp = ListDockerImageBatchJobBundleResponse.parse_raw(response.response.data)

        return resp

    def create_docker_image_batch_job(
        self,
        *,
        labels: Dict[str, str],
        docker_image_batch_job_bundle: Optional[Union[str, DockerImageBatchJobBundleResponse]] = None,
        docker_image_batch_job_bundle_name: Optional[str] = None,
        job_config: Optional[Dict[str, Any]] = None,
        cpus: Optional[int] = None,
        memory: Optional[str] = None,
        gpus: Optional[int] = None,
        gpu_type: Optional[str] = None,
        storage: Optional[str] = None,
    ):
        """
        For self hosted mode only.
        Parameters:
            docker_image_batch_job_bundle: Specifies the docker image bundle to use for the batch job.
                Either the string id of a docker image bundle, or a
                DockerImageBatchJobBundleResponse object.
                Only one of docker_image_batch_job_bundle and docker_image_batch_job_bundle_name
                can be specified.
            docker_image_batch_job_bundle_name: The name of a batch job bundle. If specified,
                Launch will use the most recent bundle with that name owned by the current user.
                Only one of docker_image_batch_job_bundle and docker_image_batch_job_bundle_name
                can be specified.
            labels: Kubernetes labels that are present on the batch job.
            job_config: A JSON-serializable python object that will get passed to the batch job,
                specifically as the contents of a file mounted at `mount_location` inside the bundle.
                You can call python's `json.load()` on the file to retrieve the contents.
            cpus: Optional override for the number of cpus to give to your job. Either the default
                must be specified in the bundle, or this must be specified.
            memory: Optional override for the amount of memory to give to your job. Either the default
                must be specified in the bundle, or this must be specified.
            gpus: Optional number of gpus to give to the bundle. If not specified in the bundle or
                here, will be interpreted as 0 gpus.
            gpu_type: Optional type of gpu. If the final number of gpus is positive, must be specified
                either in the bundle or here.
            storage: Optional reserved amount of disk to give to your batch job. If not specified,
                your job may be evicted if it is using too much disk.
        """

        assert (docker_image_batch_job_bundle is None) ^ (
            docker_image_batch_job_bundle_name is None
        ), "Exactly one of docker_image_batch_job_bundle and docker_image_batch_job_bundle_name must be specified"

        if docker_image_batch_job_bundle is not None and isinstance(
            docker_image_batch_job_bundle, DockerImageBatchJobBundleResponse
        ):
            docker_image_batch_job_bundle_id: Optional[str] = docker_image_batch_job_bundle.id
        else:
            docker_image_batch_job_bundle_id = docker_image_batch_job_bundle

        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            # Raw dictionary since it's not the entire request and we can get away with this
            # Also having it be a CreateDockerImageBatchJobResourceRequest runs into problems
            # if no values are specified
            resource_requests = dict_not_none(
                cpus=cpus,
                memory=memory,
                gpus=gpus,
                gpu_type=gpu_type,
                storage=storage,
            )
            create_docker_image_batch_job_request = CreateDockerImageBatchJobV1Request(
                **dict_not_none(
                    docker_image_batch_job_bundle_id=docker_image_batch_job_bundle_id,
                    docker_image_batch_job_bundle_name=docker_image_batch_job_bundle_name,
                    job_config=job_config,
                    labels=labels,
                    resource_requests=resource_requests,
                )
            )
            response = api_instance.create_docker_image_batch_job_v1_docker_image_batch_jobs_post(
                body=create_docker_image_batch_job_request,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
        return resp

    def get_docker_image_batch_job(self, batch_job_id: str):
        """
        For self hosted mode only. Gets information about a batch job given a batch job id.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            path_params = frozendict({"batch_job_id": batch_job_id})
            response = (
                api_instance.get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get(  # type: ignore
                    path_params=path_params,
                    skip_deserialization=True,
                )
            )
            resp = json.loads(response.response.data)

        return resp

    def update_docker_image_batch_job(self, batch_job_id: str, cancel: bool):
        """
        For self hosted mode only. Updates a batch job by id.
        Use this if you want to cancel/delete a batch job.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            path_params = frozendict({"batch_job_id": batch_job_id})
            body = UpdateDockerImageBatchJobV1Request(cancel=cancel)
            response = api_instance.update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put(  # type: ignore # noqa: E501
                body=body,
                path_params=path_params,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)

        return resp

    def create_llm_model_endpoint(
        self,
        endpoint_name: str,
        # LLM specific fields
        model_name: str,
        inference_framework_image_tag: str,
        source: LLMSource = LLMSource.HUGGING_FACE,
        inference_framework: LLMInferenceFramework = LLMInferenceFramework.DEEPSPEED,
        num_shards: int = 4,
        # General endpoint fields
        cpus: int = 32,
        memory: str = "192Gi",
        storage: Optional[str] = None,
        gpus: int = 4,
        min_workers: int = 0,
        max_workers: int = 1,
        per_worker: int = 10,
        gpu_type: Optional[str] = "nvidia-ampere-a10",
        endpoint_type: str = "sync",
        high_priority: Optional[bool] = False,
        post_inference_hooks: Optional[List[PostInferenceHooks]] = None,
        default_callback_url: Optional[str] = None,
        default_callback_auth_kind: Optional[Literal["basic", "mtls"]] = None,
        default_callback_auth_username: Optional[str] = None,
        default_callback_auth_password: Optional[str] = None,
        default_callback_auth_cert: Optional[str] = None,
        default_callback_auth_key: Optional[str] = None,
        public_inference: Optional[bool] = None,
        update_if_exists: bool = False,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Creates and registers a model endpoint in Scale Launch. The returned object is an
        instance of type ``Endpoint``, which is a base class of either ``SyncEndpoint`` or
        ``AsyncEndpoint``. This is the object to which you sent inference requests.

        Parameters:
            endpoint_name: The name of the model endpoint you want to create. The name
                must be unique across all endpoints that you own.

            model_name: name for the LLM. List can be found at
                (TODO: add list of supported models)

            inference_framework_image_tag: image tag for the inference framework.
                (TODO: use latest image tag when unspecified)

            source: source of the LLM. Currently only HuggingFace is supported.

            inference_framework: inference framework for the LLM. Currently only DeepSpeed is supported.

            num_shards: number of shards for the LLM. When bigger than 1, LLM will be sharded
                to multiple GPUs. Number of GPUs must be larger than num_shards.

            cpus: Number of cpus each worker should get, e.g. 1, 2, etc. This must be greater
                than or equal to 1.

            memory: Amount of memory each worker should get, e.g. "4Gi", "512Mi", etc. This must
                be a positive amount of memory.

            storage: Amount of local ephemeral storage each worker should get, e.g. "4Gi",
                "512Mi", etc. This must be a positive amount of storage.

            gpus: Number of gpus each worker should get, e.g. 0, 1, etc.

            min_workers: The minimum number of workers. Must be greater than or equal to 0. This
                should be determined by computing the minimum throughput of your workload and
                dividing it by the throughput of a single worker. This field must be at least ``1``
                for synchronous endpoints.

            max_workers: The maximum number of workers. Must be greater than or equal to 0,
                and as well as greater than or equal to ``min_workers``. This should be determined by
                computing the maximum throughput of your workload and dividing it by the throughput
                of a single worker.

            per_worker: The maximum number of concurrent requests that an individual worker can
                service. Launch automatically scales the number of workers for the endpoint so that
                each worker is processing ``per_worker`` requests, subject to the limits defined by
                ``min_workers`` and ``max_workers``.

                - If the average number of concurrent requests per worker is lower than
                ``per_worker``, then the number of workers will be reduced. - Otherwise,
                if the average number of concurrent requests per worker is higher than
                ``per_worker``, then the number of workers will be increased to meet the elevated
                traffic.

                Here is our recommendation for computing ``per_worker``:

                1. Compute ``min_workers`` and ``max_workers`` per your minimum and maximum
                throughput requirements. 2. Determine a value for the maximum number of
                concurrent requests in the workload. Divide this number by ``max_workers``. Doing
                this ensures that the number of workers will "climb" to ``max_workers``.

            gpu_type: If specifying a non-zero number of gpus, this controls the type of gpu
                requested. Here are the supported values:

                - ``nvidia-tesla-t4``
                - ``nvidia-ampere-a10``

            endpoint_type: Either ``"sync"`` or ``"async"``.

            high_priority: Either ``True`` or ``False``. Enabling this will allow the created
                endpoint to leverage the shared pool of prewarmed nodes for faster spinup time.

            post_inference_hooks: List of hooks to trigger after inference tasks are served.

            default_callback_url: The default callback url to use for async endpoints.
                This can be overridden in the task parameters for each individual task.
                post_inference_hooks must contain "callback" for the callback to be triggered.

            default_callback_auth_kind: The default callback auth kind to use for async endpoints.
                Either "basic" or "mtls". This can be overridden in the task parameters for each
                individual task.

            default_callback_auth_username: The default callback auth username to use. This only
                applies if default_callback_auth_kind is "basic". This can be overridden in the task
                parameters for each individual task.

            default_callback_auth_password: The default callback auth password to use. This only
                applies if default_callback_auth_kind is "basic". This can be overridden in the task
                parameters for each individual task.

            default_callback_auth_cert: The default callback auth cert to use. This only applies
                if default_callback_auth_kind is "mtls". This can be overridden in the task
                parameters for each individual task.

            default_callback_auth_key: The default callback auth key to use. This only applies
                if default_callback_auth_kind is "mtls". This can be overridden in the task
                parameters for each individual task.

            public_inference: If ``True``, this endpoint will be available to all user IDs for
                inference.

            update_if_exists: If ``True``, will attempt to update the endpoint if it exists.
                Otherwise, will unconditionally try to create a new endpoint. Note that endpoint
                names for a given user must be unique, so attempting to call this function with
                ``update_if_exists=False`` for an existing endpoint will raise an error.

            labels: An optional dictionary of key/value pairs to associate with this endpoint.

        Returns:
            A Endpoint object that can be used to make requests to the endpoint.

        """
        existing_endpoint = self.get_model_endpoint(endpoint_name)
        if update_if_exists and existing_endpoint:
            self.edit_model_endpoint(
                model_endpoint=endpoint_name,
                model_bundle=existing_endpoint.model_endpoint.bundle_name,
                cpus=cpus,
                memory=memory,
                storage=storage,
                gpus=gpus,
                min_workers=min_workers,
                max_workers=max_workers,
                per_worker=per_worker,
                gpu_type=gpu_type,
                high_priority=high_priority,
                default_callback_url=default_callback_url,
                default_callback_auth_kind=default_callback_auth_kind,
                default_callback_auth_username=default_callback_auth_username,
                default_callback_auth_password=default_callback_auth_password,
                default_callback_auth_cert=default_callback_auth_cert,
                default_callback_auth_key=default_callback_auth_key,
                public_inference=public_inference,
            )
            return existing_endpoint
        else:
            # Presumably, the user knows that the endpoint doesn't already exist, and so we can
            # defer to the server to reject any duplicate creations.
            logger.info("Creating new LLM endpoint")
            with ApiClient(self.configuration) as api_client:
                api_instance = DefaultApi(api_client)
                post_inference_hooks_strs = None
                if post_inference_hooks is not None:
                    post_inference_hooks_strs = []
                    for hook in post_inference_hooks:
                        if isinstance(hook, PostInferenceHooks):
                            post_inference_hooks_strs.append(hook.value)
                        else:
                            post_inference_hooks_strs.append(hook)

                if default_callback_auth_kind is not None:
                    default_callback_auth = CallbackAuth(
                        **dict_not_none(
                            kind=default_callback_auth_kind,
                            username=default_callback_auth_username,
                            password=default_callback_auth_password,
                            cert=default_callback_auth_cert,
                            key=default_callback_auth_key,
                        )
                    )
                else:
                    default_callback_auth = None

                payload = dict_not_none(
                    name=endpoint_name,
                    model_name=model_name,
                    source=source,
                    inference_framework=inference_framework,
                    inference_framework_image_tag=inference_framework_image_tag,
                    num_shards=num_shards,
                    cpus=cpus,
                    endpoint_type=ModelEndpointType(endpoint_type),
                    gpus=gpus,
                    gpu_type=GpuType(gpu_type) if gpu_type is not None else None,
                    labels=labels or {},
                    max_workers=max_workers,
                    memory=memory,
                    metadata={},
                    min_workers=min_workers,
                    per_worker=per_worker,
                    high_priority=high_priority,
                    post_inference_hooks=post_inference_hooks_strs,
                    default_callback_url=default_callback_url,
                    default_callback_auth=default_callback_auth,
                    storage=storage,
                    public_inference=public_inference,
                )
                create_model_endpoint_request = CreateLLMModelEndpointV1Request(**payload)
                response = api_instance.create_model_endpoint_v1_llm_model_endpoints_post(
                    body=create_model_endpoint_request,
                    skip_deserialization=True,
                )
                resp = json.loads(response.response.data)
            endpoint_creation_task_id = resp.get("endpoint_creation_task_id", None)  # TODO probably throw on None
            logger.info("Endpoint creation task id is %s", endpoint_creation_task_id)
            model_endpoint = ModelEndpoint(
                name=endpoint_name, bundle_name=f"{endpoint_name}-{str(inference_framework)}"
            )
            if endpoint_type == "async":
                return AsyncEndpoint(model_endpoint=model_endpoint, client=self)
            elif endpoint_type == "sync":
                return SyncEndpoint(model_endpoint=model_endpoint, client=self)
            else:
                raise ValueError("Endpoint should be one of the types 'sync' or 'async'")

    def list_llm_model_endpoints(self) -> List[Endpoint]:
        """
        Lists all LLM model endpoints that the user has access to.

        Returns:
            A list of ``ModelEndpoint`` objects.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            response = api_instance.list_model_endpoints_v1_llm_model_endpoints_get(skip_deserialization=True)
            resp = json.loads(response.response.data)
        async_endpoints: List[Endpoint] = [
            AsyncEndpoint(
                model_endpoint=ModelEndpoint.from_dict(endpoint),  # type: ignore
                client=self,
            )
            for endpoint in resp["model_endpoints"]
            if endpoint["spec"]["endpoint_type"] == "async"
        ]
        sync_endpoints: List[Endpoint] = [
            SyncEndpoint(
                model_endpoint=ModelEndpoint.from_dict(endpoint),  # type: ignore
                client=self,
            )
            for endpoint in resp["model_endpoints"]
            if endpoint["spec"]["endpoint_type"] == "sync"
        ]
        streaming_endpoints: List[Endpoint] = [
            StreamingEndpoint(
                model_endpoint=ModelEndpoint.from_dict(endpoint),  # type: ignore
                client=self,
            )
            for endpoint in resp["model_endpoints"]
            if endpoint["spec"]["endpoint_type"] == "streaming"
        ]
        return async_endpoints + sync_endpoints + streaming_endpoints

    def get_llm_model_endpoint(
        self, endpoint_name: str
    ) -> Optional[Union[AsyncEndpoint, SyncEndpoint, StreamingEndpoint]]:
        """
        Gets a model endpoint associated with a name that the user has access to.

        Parameters:
            endpoint_name: The name of the endpoint to retrieve.
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            path_params = frozendict({"model_endpoint_name": endpoint_name})
            response = api_instance.get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get(  # type: ignore
                path_params=path_params,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)

        if resp["spec"]["endpoint_type"] == "async":
            return AsyncEndpoint(ModelEndpoint.from_dict(resp), client=self)  # type: ignore
        elif resp["spec"]["endpoint_type"] == "sync":
            return SyncEndpoint(ModelEndpoint.from_dict(resp), client=self)  # type: ignore
        elif resp["spec"]["endpoint_type"] == "streaming":
            return StreamingEndpoint(ModelEndpoint.from_dict(resp), client=self)  # type: ignore
        else:
            raise ValueError("Endpoint should be one of the types 'sync', 'async', or 'streaming'")

    def completions_sync(
        self,
        endpoint_name: str,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
    ) -> CompletionSyncV1Response:
        """
        Run prompt completion on a sync LLM endpoint. Will fail if the endpoint is not sync.

        Parameters:
            endpoint_name: The name of the LLM endpoint to make the request to

            prompts: The list of prompts to send to the endpoint

            max_new_tokens: The maximum number of tokens to generate for each prompt

            temperature: The temperature to use for sampling

        Returns:
            Response for prompt completion
        """
        with ApiClient(self.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            request = CompletionSyncV1Request(max_new_tokens=max_new_tokens, prompts=prompts, temperature=temperature)
            query_params = frozendict({"model_endpoint_name": endpoint_name})
            response = api_instance.create_completion_sync_task_v1_llm_completions_sync_post(  # type: ignore
                body=request,
                query_params=query_params,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
        return resp


def _zip_directory(zipf: ZipFile, path: str) -> None:
    for root, _, files in os.walk(path):
        for file_ in files:
            zipf.write(
                filename=os.path.join(root, file_),
                arcname=os.path.relpath(os.path.join(root, file_), os.path.join(path, "..")),
            )


def _zip_directories(zip_path: str, dir_list: List[str]) -> None:
    with ZipFile(zip_path, "w") as zip_f:
        for dir_ in dir_list:
            _zip_directory(zip_f, dir_)
