import inspect
import logging
import os
import shutil
import tempfile
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import cloudpickle
import requests
import yaml

from launch.connection import Connection
from launch.constants import (
    ASYNC_TASK_PATH,
    ASYNC_TASK_RESULT_PATH,
    BATCH_TASK_PATH,
    BATCH_TASK_RESULTS_PATH,
    ENDPOINT_PATH,
    MODEL_BUNDLE_SIGNED_URL_PATH,
    SCALE_LAUNCH_ENDPOINT,
    SYNC_TASK_PATH,
)
from launch.errors import APIError
from launch.find_packages import find_packages_from_imports, get_imports
from launch.make_batch_file import make_batch_input_file
from launch.model_bundle import ModelBundle
from launch.model_endpoint import (
    AsyncEndpoint,
    Endpoint,
    ModelEndpoint,
    SyncEndpoint,
)
from launch.request_validation import validate_task_request
from launch.utils import trim_kwargs

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


def _add_app_config_to_bundle_create_payload(
    payload: Dict[str, Any], app_config: Optional[Union[Dict[str, Any], str]]
):
    """
    Edits a request payload (for creating a bundle) to include a (not serialized) app_config if it's not None
    """
    if isinstance(app_config, Dict):
        payload["app_config"] = app_config
    elif isinstance(app_config, str):
        with open(  # pylint: disable=unspecified-encoding
            app_config, "r"
        ) as f:
            app_config_dict = yaml.safe_load(f)
            payload["app_config"] = app_config_dict


class LaunchClient:
    """Scale Launch Python Client extension."""

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
        endpoint = endpoint or SCALE_LAUNCH_ENDPOINT
        self.connection = Connection(api_key, endpoint)
        self.self_hosted = self_hosted
        self.upload_bundle_fn: Optional[Callable[[str, str], None]] = None
        self.upload_batch_csv_fn: Optional[Callable[[str, str], None]] = None
        self.endpoint_auth_decorator_fn: Callable[
            [Dict[str, Any]], Dict[str, Any]
        ] = lambda x: x
        self.bundle_location_fn: Optional[Callable[[], str]] = None

    def __repr__(self):
        return f"LaunchClient(connection='{self.connection}')"

    def __eq__(self, other):
        return self.connection == other.connection

    def register_upload_bundle_fn(
        self, upload_bundle_fn: Callable[[str, str], None]
    ):
        """
        For self-hosted mode only. Registers a function that handles model bundle upload. This function is called as

        upload_bundle_fn(serialized_bundle, bundle_url)

        This function should directly write the contents of serialized_bundle as a binary string into bundle_url.

        See register_bundle_location_fn for more notes on the signature of upload_bundle_fn

        Parameters:
            upload_bundle_fn: Function that takes in a serialized bundle (bytes type), and uploads that bundle to an appropriate
                location. Only needed for self-hosted mode.
        """
        self.upload_bundle_fn = upload_bundle_fn

    def register_upload_batch_csv_fn(
        self, upload_batch_csv_fn: Callable[[str, str], None]
    ):
        """
        For self-hosted mode only. Registers a function that handles batch text upload. This function is called as

        upload_batch_csv_fn(csv_text, csv_url)

        This function should directly write the contents of csv_text as a text string into csv_url.

        Parameters:
            upload_batch_csv_fn: Function that takes in a csv text (string type), and uploads that bundle to an appropriate
                location. Only needed for self-hosted mode.
        """
        self.upload_batch_csv_fn = upload_batch_csv_fn

    def register_bundle_location_fn(
        self, bundle_location_fn: Callable[[], str]
    ):
        """
        For self-hosted mode only. Registers a function that gives a location for a model bundle. Should give different
        locations each time. This function is called as bundle_location_fn(), and should return a bundle_url that
        register_upload_bundle_fn can take.

        Strictly, bundle_location_fn() does not need to return a str. The only requirement is that if bundle_location_fn
        returns a value of type T, then upload_bundle_fn() takes in an object of type T as its second argument
        (i.e. bundle_url).

        Parameters:
            bundle_location_fn: Function that generates bundle_urls for upload_bundle_fn.
        """
        self.bundle_location_fn = bundle_location_fn

    def register_endpoint_auth_decorator(self, endpoint_auth_decorator_fn):
        """
        For self-hosted mode only. Registers a function that modifies the endpoint creation payload to include
        required fields for self-hosting.
        """
        self.endpoint_auth_decorator_fn = endpoint_auth_decorator_fn

    def create_model_bundle_from_dir(
        self,
        model_bundle_name: str,
        base_path: str,
        requirements_path: str,
        env_params: Dict[str, str],
        load_predict_fn_module_path: str,
        load_model_fn_module_path: str,
        app_config: Optional[Union[Dict[str, Any], str]] = None,
    ) -> ModelBundle:
        """
        Packages up code from a local filesystem folder and uploads that as a bundle to Scale Launch.
        In this mode, a bundle is just local code instead of a serialized object.

        Parameters:
            model_bundle_name: Name of model bundle you want to create. This acts as a unique identifier.
            base_path: The path on the local filesystem where the bundle code lives.
            requirements_path: A path on the local filesystem where a requirements.txt file lives.
            env_params: A dictionary that dictates environment information e.g.
                the use of pytorch or tensorflow, which cuda/cudnn versions to use.
                Specifically, the dictionary should contain the following keys:
                "framework_type": either "tensorflow" or "pytorch".
                "pytorch_version": Version of pytorch, e.g. "1.5.1", "1.7.0", etc. Only applicable if framework_type is pytorch
                "cuda_version": Version of cuda used, e.g. "11.0".
                "cudnn_version" Version of cudnn used, e.g. "cudnn8-devel".
                "tensorflow_version": Version of tensorflow, e.g. "2.3.0". Only applicable if framework_type is tensorflow
            load_predict_fn_module_path: A python module path within base_path for a function that, when called with the output of
                load_model_fn_module_path, returns a function that carries out inference.
            load_model_fn_module_path: A python module path within base_path for a function that returns a model. The output feeds into
                the function located at load_predict_fn_module_path.
            app_config: Either a Dictionary that represents a YAML file contents or a local path to a YAML file.
        """
        with open(requirements_path, "r", encoding="utf-8") as req_f:
            requirements = req_f.read().splitlines()

        tmpdir = tempfile.mkdtemp()
        try:
            tmparchive = os.path.join(tmpdir, "bundle")
            abs_base_path = os.path.abspath(base_path)
            root_dir = os.path.dirname(abs_base_path)
            base_dir = os.path.basename(abs_base_path)

            with open(
                shutil.make_archive(
                    base_name=tmparchive,
                    format="zip",
                    root_dir=root_dir,
                    base_dir=base_dir,
                ),
                "rb",
            ) as zip_f:
                data = zip_f.read()
        finally:
            shutil.rmtree(tmpdir)

        if self.self_hosted:
            if self.upload_bundle_fn is None:
                raise ValueError("Upload_bundle_fn should be registered")
            if self.bundle_location_fn is None:
                raise ValueError(
                    "Need either bundle_location_fn to know where to upload bundles"
                )
            raw_bundle_url = self.bundle_location_fn()  # type: ignore
            self.upload_bundle_fn(data, raw_bundle_url)  # type: ignore
        else:
            model_bundle_url = self.connection.post(
                {}, MODEL_BUNDLE_SIGNED_URL_PATH
            )
            s3_path = model_bundle_url["signedUrl"]
            raw_bundle_url = (
                f"s3://{model_bundle_url['bucket']}/{model_bundle_url['key']}"
            )
            requests.put(s3_path, data=data)

        bundle_metadata = {
            "load_predict_fn_module_path": load_predict_fn_module_path,
            "load_model_fn_module_path": load_model_fn_module_path,
            "base_dir": base_dir,
        }

        logger.info(
            "create_model_bundle_from_dir: raw_bundle_url=%s",
            raw_bundle_url,
        )
        payload = dict(
            packaging_type="zip",
            bundle_name=model_bundle_name,
            location=raw_bundle_url,
            bundle_metadata=bundle_metadata,
            requirements=requirements,
            env_params=env_params,
        )
        _add_app_config_to_bundle_create_payload(payload, app_config)

        self.connection.post(
            payload=payload,
            route="model_bundle",
        )
        return ModelBundle(model_bundle_name)

    def create_model_bundle(  # pylint: disable=too-many-statements
        self,
        model_bundle_name: str,
        env_params: Dict[str, str],
        *,
        load_predict_fn: Optional[
            Callable[[LaunchModel_T], Callable[[Any], Any]]
        ] = None,
        predict_fn_or_cls: Optional[Callable[[Any], Any]] = None,
        requirements: Optional[List[str]] = None,
        model: Optional[LaunchModel_T] = None,
        load_model_fn: Optional[Callable[[], LaunchModel_T]] = None,
        bundle_url: Optional[str] = None,
        app_config: Optional[Union[Dict[str, Any], str]] = None,
        globals_copy: Optional[Dict[str, Any]] = None,
    ) -> ModelBundle:
        """
        Grabs a s3 signed url and uploads a model bundle to Scale Launch.

        A model bundle consists of exactly {predict_fn_or_cls}, {load_predict_fn + model}, or {load_predict_fn + load_model_fn}.
        Pre/post-processing code can be included inside load_predict_fn/model or in predict_fn_or_cls call.

        Parameters:
            model_bundle_name: Name of model bundle you want to create. This acts as a unique identifier.
            predict_fn_or_cls: Function or a Callable class that runs end-to-end (pre/post processing and model inference) on the call.
                I.e. `predict_fn_or_cls(REQUEST) -> RESPONSE`.
            model: Typically a trained Neural Network, e.g. a Pytorch module
            load_predict_fn: Function that when called with model, returns a function that carries out inference
                I.e. `load_predict_fn(model) -> func; func(REQUEST) -> RESPONSE`
            load_model_fn: Function that when run, loads a model, e.g. a Pytorch module
                I.e. `load_predict_fn(load_model_fn()) -> func; func(REQUEST) -> RESPONSE`
            bundle_url: Only for self-hosted mode. Desired location of bundle.
            Overrides any value given by self.bundle_location_fn
            requirements: A list of python package requirements, e.g.
                ["tensorflow==2.3.0", "tensorflow-hub==0.11.0"]. If no list has been passed, will default to the currently
                imported list of packages.
            app_config: Either a Dictionary that represents a YAML file contents or a local path to a YAML file.
            env_params: A dictionary that dictates environment information e.g.
                the use of pytorch or tensorflow, which cuda/cudnn versions to use.
                Specifically, the dictionary should contain the following keys:
                "framework_type": either "tensorflow" or "pytorch".
                "pytorch_version": Version of pytorch, e.g. "1.5.1", "1.7.0", etc. Only applicable if framework_type is pytorch
                "cuda_version": Version of cuda used, e.g. "11.0".
                "cudnn_version" Version of cudnn used, e.g. "cudnn8-devel".
                "tensorflow_version": Version of tensorflow, e.g. "2.3.0". Only applicable if framework_type is tensorflow
            globals_copy: Dictionary of the global symbol table. Normally provided by `globals()` built-in function.
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
                "A model bundle consists of exactly {predict_fn_or_cls}, {load_predict_fn + model}, or {load_predict_fn + load_model_fn}."
            )
        # TODO should we try to catch when people intentionally pass both model and load_model_fn as None?

        if requirements is None:
            # TODO explore: does globals() actually work as expected? Should we use globals_copy instead?
            requirements_inferred = find_packages_from_imports(globals())
            requirements = [
                f"{key}=={value}"
                for key, value in requirements_inferred.items()
            ]
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

        bundle: Union[
            Callable[[Any], Any], Dict[str, Any], None
        ]  # validate bundle
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
            bundle_metadata["load_predict_fn"] = inspect.getsource(
                load_predict_fn  # type: ignore
            )
        else:
            bundle = dict(
                load_model_fn=load_model_fn, load_predict_fn=load_predict_fn
            )
            bundle_metadata["load_predict_fn"] = inspect.getsource(
                load_predict_fn  # type: ignore
            )
            bundle_metadata["load_model_fn"] = inspect.getsource(
                load_model_fn  # type: ignore
            )

        serialized_bundle = cloudpickle.dumps(bundle)

        if self.self_hosted:
            if self.upload_bundle_fn is None:
                raise ValueError("Upload_bundle_fn should be registered")
            if self.bundle_location_fn is None and bundle_url is None:
                raise ValueError(
                    "Need either bundle_location_fn or bundle_url to know where to upload bundles"
                )
            if bundle_url is None:
                bundle_url = self.bundle_location_fn()  # type: ignore
            self.upload_bundle_fn(serialized_bundle, bundle_url)
            raw_bundle_url = bundle_url
        else:
            # Grab a signed url to make upload to
            model_bundle_s3_url = self.connection.post(
                {}, MODEL_BUNDLE_SIGNED_URL_PATH
            )
            s3_path = model_bundle_s3_url["signedUrl"]
            raw_bundle_url = f"s3://{model_bundle_s3_url['bucket']}/{model_bundle_s3_url['key']}"

            # Make bundle upload

            requests.put(s3_path, data=serialized_bundle)

        payload = dict(
            packaging_type="cloudpickle",
            bundle_name=model_bundle_name,
            location=raw_bundle_url,
            bundle_metadata=bundle_metadata,
            requirements=requirements,
            env_params=env_params,
        )

        _add_app_config_to_bundle_create_payload(payload, app_config)

        self.connection.post(
            payload=payload,
            route="model_bundle",
        )  # TODO use return value somehow
        # resp["data"]["bundle_name"] should equal model_bundle_name
        # TODO check that a model bundle was created and no name collisions happened
        return ModelBundle(model_bundle_name)

    def create_model_endpoint(
        self,
        endpoint_name: str,
        model_bundle: Union[ModelBundle, str],
        cpus: int = 3,
        memory: str = "8Gi",
        gpus: int = 0,
        min_workers: int = 1,
        max_workers: int = 1,
        per_worker: int = 1,
        gpu_type: Optional[str] = None,
        endpoint_type: str = "sync",
        update_if_exists: bool = False,
    ) -> Optional[Endpoint]:
        """
        Creates a Model Endpoint that is able to serve requests.
        Corresponds to POST/PUT endpoints

        Parameters:
            endpoint_name: Name of model endpoint. Must be unique.
            model_bundle: The ModelBundle that you want your Model Endpoint to serve
            cpus: Number of cpus each worker should get, e.g. 1, 2, etc.
            memory: Amount of memory each worker should get, e.g. "4Gi", "512Mi", etc.
            gpus: Number of gpus each worker should get, e.g. 0, 1, etc.
            min_workers: Minimum number of workers for model endpoint
            max_workers: Maximum number of workers for model endpoint
            per_worker: An autoscaling parameter. Use this to make a tradeoff between latency and costs,
                a lower per_worker will mean more workers are created for a given workload
            gpu_type: If specifying a non-zero number of gpus, this controls the type of gpu requested. Current options are
                "nvidia-tesla-t4" for NVIDIA T4s, or "nvidia-tesla-v100" for NVIDIA V100s.
            endpoint_type: Either "sync" or "async". Type of endpoint we want to instantiate.

        Returns:
             A Endpoint object that can be used to make requests to the endpoint.

        """
        if (
            update_if_exists
            and self.get_model_endpoint(endpoint_name) is not None
        ):
            self.edit_model_endpoint(
                endpoint_name=endpoint_name,
                model_bundle=model_bundle,
                cpus=cpus,
                memory=memory,
                gpus=gpus,
                min_workers=min_workers,
                max_workers=max_workers,
                per_worker=per_worker,
                gpu_type=gpu_type,
            )
            # R1710: Either all return statements in a function should return an expression, or none of them should.
            return None
        else:
            # Presumably, the user knows that the endpoint doesn't already exist, and so we can defer
            # to the server to reject any duplicate creations.
            logger.info("Creating new endpoint")
            payload = dict(
                endpoint_name=endpoint_name,
                bundle_name=_model_bundle_to_name(model_bundle),
                cpus=cpus,
                memory=memory,
                gpus=gpus,
                gpu_type=gpu_type,
                min_workers=min_workers,
                max_workers=max_workers,
                per_worker=per_worker,
                endpoint_type=endpoint_type,
            )
            if gpus == 0:
                del payload["gpu_type"]
            elif gpus > 0 and gpu_type is None:
                raise ValueError("If nonzero gpus, must provide gpu_type")
            payload = self.endpoint_auth_decorator_fn(payload)
            resp = self.connection.post(payload, ENDPOINT_PATH)
            endpoint_creation_task_id = resp.get(
                "endpoint_creation_task_id", None
            )  # TODO probably throw on None
            logger.info(
                "Endpoint creation task id is %s", endpoint_creation_task_id
            )
            model_endpoint = ModelEndpoint(name=endpoint_name)
            if endpoint_type == "async":
                return AsyncEndpoint(
                    model_endpoint=model_endpoint, client=self
                )
            elif endpoint_type == "sync":
                return SyncEndpoint(model_endpoint=model_endpoint, client=self)
            else:
                raise ValueError(
                    "Endpoint should be one of the types 'sync' or 'async'"
                )

    def edit_model_endpoint(
        self,
        endpoint_name: str,
        model_bundle: Optional[Union[ModelBundle, str]] = None,
        cpus: Optional[float] = None,
        memory: Optional[str] = None,
        gpus: Optional[int] = None,
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        per_worker: Optional[int] = None,
        gpu_type: Optional[str] = None,
    ) -> None:
        """
        Edit an existing model endpoint
        """
        logger.info("Editing existing endpoint")
        bundle_name = (
            _model_bundle_to_name(model_bundle) if model_bundle else None
        )
        payload = dict(
            bundle_name=bundle_name,
            cpus=cpus,
            memory=memory,
            gpus=gpus,
            gpu_type=gpu_type,
            min_workers=min_workers,
            max_workers=max_workers,
            per_worker=per_worker,
        )
        # Allows changing some authorization settings by changing endpoint_auth_decorator_fn
        payload = self.endpoint_auth_decorator_fn(payload)
        if gpus == 0 and gpu_type is not None:
            logger.warning("GPU type setting %s will have no effect", gpu_type)
            payload["gpu_type"] = None
        payload = trim_kwargs(payload)
        resp = self.connection.put(payload, f"{ENDPOINT_PATH}/{endpoint_name}")
        endpoint_creation_task_id = resp.get(
            "endpoint_creation_task_id", None
        )  # Returned from server as "creation"
        logger.info("Endpoint edit task id is %s", endpoint_creation_task_id)

    def get_model_endpoint(
        self, endpoint_name: str
    ) -> Optional[Union[AsyncEndpoint, SyncEndpoint]]:
        try:
            resp = self.connection.get(
                os.path.join(ENDPOINT_PATH, endpoint_name)
            )
        except APIError:
            logger.exception(
                "Got an error when retrieving endpoint %s", endpoint_name
            )
            return None

        if resp["endpoint_type"] == "async":
            return AsyncEndpoint(
                ModelEndpoint(name=resp["endpoint_name"]), client=self
            )
        elif resp["endpoint_type"] == "sync":
            return SyncEndpoint(
                ModelEndpoint(name=resp["endpoint_name"]), client=self
            )
        else:
            raise ValueError(
                "Endpoint should be one of the types 'sync' or 'async'"
            )

    # Relatively small wrappers around http requests

    def list_model_bundles(self) -> List[ModelBundle]:
        """
        Returns a list of model bundles that the user owns.

        Returns:
            A list of ModelBundle objects
        """
        resp = self.connection.get("model_bundle")
        model_bundles = [
            ModelBundle.from_dict(item) for item in resp["bundles"]  # type: ignore
        ]
        return model_bundles

    def get_model_bundle(self, bundle_name: str) -> ModelBundle:
        """
        Returns a Model Bundle object specified by `bundle_name`.
        Returns:
            A ModelBundle object
        """
        resp = self.connection.get(f"model_bundle/{bundle_name}")
        assert (
            len(resp["bundles"]) == 1
        ), f"Bundle with name `{bundle_name}` not found"
        return ModelBundle.from_dict(resp["bundles"][0])  # type: ignore

    def list_model_endpoints(
        self,
    ) -> List[Endpoint]:
        """
        Lists all model endpoints that the user owns.
        TODO: single get_model_endpoint(self)? route doesn't exist serverside I think

        Returns:
            A list of ModelEndpoint objects
        """
        resp = self.connection.get(ENDPOINT_PATH)
        async_endpoints: List[Endpoint] = [
            AsyncEndpoint(
                model_endpoint=ModelEndpoint.from_dict(endpoint),  # type: ignore
                client=self,
            )
            for endpoint in resp["endpoints"]
            if endpoint["endpoint_type"] == "async"
        ]
        sync_endpoints: List[Endpoint] = [
            SyncEndpoint(
                model_endpoint=ModelEndpoint.from_dict(endpoint), client=self  # type: ignore
            )
            for endpoint in resp["endpoints"]
            if endpoint["endpoint_type"] == "sync"
        ]
        return async_endpoints + sync_endpoints

    def delete_model_bundle(self, model_bundle: ModelBundle):
        """
        Deletes the model bundle on the server.
        """
        route = f"model_bundle/{model_bundle.name}"
        resp = self.connection.delete(route)
        return resp["deleted"]

    def delete_model_endpoint(self, model_endpoint: ModelEndpoint):
        """
        Deletes a model endpoint.
        """
        route = f"{ENDPOINT_PATH}/{model_endpoint.name}"
        resp = self.connection.delete(route)
        return resp["deleted"]

    def read_endpoint_creation_logs(self, endpoint_name: str):
        """
        Get builder logs as text.
        """
        route = f"{ENDPOINT_PATH}/creation_logs/{endpoint_name}"
        resp = self.connection.get(route)
        return resp["content"]

    def sync_request(
        self,
        endpoint_id: str,
        url: Optional[str] = None,
        args: Optional[Dict] = None,
        return_pickled: bool = True,
    ) -> Dict[str, Any]:
        """
        Not recommended for use, instead use functions provided by SyncEndpoint
        Makes a request to the Sync Model Endpoint at endpoint_id, and blocks until request completion or timeout.
        Endpoint at endpoint_id must be a SyncEndpoint, otherwise this request will fail.

        Parameters:
            endpoint_id: The id of the endpoint to make the request to
            url: A url that points to a file containing model input.
                Must be accessible by Scale Launch, hence it needs to either be public or a signedURL.
            args: A dictionary of arguments to the `predict` function defined in your model bundle.
                Must be json-serializable, i.e. composed of str, int, float, etc.
                If your `predict` function has signature `predict(foo, bar)`, then args should be a dictionary with
                keys `foo` and `bar`. Exactly one of url and args must be specified.
            return_pickled: Whether the python object returned is pickled, or directly written to the file returned.

        Returns:
            A dictionary with key either "result_url" or "result", depending on the value of `return_pickled`.
            If `return_pickled` is true, the key will be "result_url",
            and the value is a signedUrl that contains a cloudpickled Python object,
            the result of running inference on the model input.
            Example output:
                `https://foo.s3.us-west-2.amazonaws.com/bar/baz/qux?xyzzy`

            Otherwise, if `return_pickled` is false, the key will be "result",
            and the value is the output of the endpoint's `predict` function, serialized as json.
        """
        validate_task_request(url=url, args=args)
        payload: Dict[str, Any] = dict(return_pickled=return_pickled)
        if url is not None:
            payload["url"] = url
        if args is not None:
            payload["args"] = args
        resp = self.connection.post(
            payload=payload,
            route=f"{SYNC_TASK_PATH}/{endpoint_id}",
        )
        return resp

    def async_request(
        self,
        endpoint_id: str,
        url: Optional[str] = None,
        args: Optional[Dict] = None,
        return_pickled: bool = True,
    ) -> str:
        """
        Not recommended to use this, instead we recommend to use functions provided by AsyncEndpoint.
        Makes a request to the Async Model Endpoint at endpoint_id, and immediately returns a key that can be used to retrieve
        the result of inference at a later time.
        Endpoint

        Parameters:
            endpoint_id: The id of the endpoint to make the request to
            url: A url that points to a file containing model input.
                Must be accessible by Scale Launch, hence it needs to either be public or a signedURL.
            args: A dictionary of arguments to the ModelBundle's predict function.
                Must be json-serializable, i.e. composed of str, int, float, etc.
                If your `predict` function has signature `predict(foo, bar)`, then args should be a dictionary with
                keys `foo` and `bar`. Exactly one of url and args must be specified.
            return_pickled: Whether the python object returned is pickled, or directly written to the file returned.

        Returns:
            An id/key that can be used to fetch inference results at a later time.
            Example output:
                `abcabcab-cabc-abca-0123456789ab`
        """
        validate_task_request(url=url, args=args)
        payload: Dict[str, Any] = dict(return_pickled=return_pickled)
        if url is not None:
            payload["url"] = url
        if args is not None:
            payload["args"] = args

        resp = self.connection.post(
            payload=payload,
            route=f"{ASYNC_TASK_PATH}/{endpoint_id}",
        )
        return resp["task_id"]

    def get_async_response(self, async_task_id: str) -> Dict[str, Any]:
        """
        Not recommended to use this, instead we recommend to use functions provided by AsyncEndpoint.
        Gets inference results from a previously created task.

        Parameters:
            async_task_id: The id/key returned from a previous invocation of async_request.

        Returns:
            A dictionary that contains task status and optionally a result url or result if the task has completed.
            Result url or result will be returned if the task has succeeded. Will return a result url iff `return_pickled`
            was set to True on task creation.
            Dictionary's keys are as follows:
            state: 'PENDING' or 'SUCCESS' or 'FAILURE'
            result_url: a url pointing to inference results. This url is accessible for 12 hours after the request has been made.
            result: the value returned by the endpoint's `predict` function, serialized as json
            Example output:
                `{'state': 'SUCCESS', 'result_url': 'https://foo.s3.us-west-2.amazonaws.com/bar/baz/qux?xyzzy'}`
        TODO: do we want to read the results from here as well? i.e. translate result_url into a python object
        """

        resp = self.connection.get(
            route=f"{ASYNC_TASK_RESULT_PATH}/{async_task_id}"
        )
        return resp

    def batch_async_request(
        self,
        bundle_name: str,
        # urls_file: str,
        urls: List[str],
        serialization_format: str = "json",
    ):
        """
        Sends a batch inference request to the Model Endpoint at endpoint_id, returns a key that can be used to retrieve
        the results of inference at a later time.

        Parameters:
            bundle_name: The id of the bundle to make the request to
            serialization_format: Serialization format of output, either 'pickle' or 'json'.
                'pickle' corresponds to pickling results + returning
            urls_file: S3 location of a CSV that is used as input to the batch job. TODO this really should be a list of urls
            urls: A list of urls, each pointing to a file containing model input.
                Must be accessible by Scale Launch, hence urls need to either be public or signedURLs.

        Returns:
            An id/key that can be used to fetch inference results at a later time
        """
        f = StringIO()
        make_batch_input_file(urls, f)
        f.seek(0)

        if self.self_hosted:
            # TODO make this not use bundle_location_fn()
            file_location = self.bundle_location_fn()  # type: ignore
            self.upload_batch_csv_fn(
                f.getvalue(), file_location
            )  # type: ignore
        else:
            model_bundle_s3_url = self.connection.post(
                {}, MODEL_BUNDLE_SIGNED_URL_PATH
            )
            s3_path = model_bundle_s3_url["signedUrl"]
            requests.put(s3_path, data=f.getvalue())
            file_location = f"s3://{model_bundle_s3_url['bucket']}/{model_bundle_s3_url['key']}"

        logger.info("Writing batch task csv to %s", file_location)

        payload = dict(
            input_path=file_location,
            serialization_format=serialization_format,
        )
        payload = self.endpoint_auth_decorator_fn(payload)
        resp = self.connection.post(
            route=f"{BATCH_TASK_PATH}/{bundle_name}",
            payload=payload,
        )
        return resp["job_id"]

    def get_batch_async_response(self, batch_async_task_id: str):
        """
        TODO not sure about how the batch task returns an identifier for the batch.
        Gets inference results from a previously created batch task.

        Parameters:
            batch_async_task_id: An id representing the batch task job

        Returns:
            TODO Something similar to a list of signed s3URLs
        """
        resp = self.connection.get(
            route=f"{BATCH_TASK_RESULTS_PATH}/{batch_async_task_id}"
        )
        return resp
