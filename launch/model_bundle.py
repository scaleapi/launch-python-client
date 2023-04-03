import datetime
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from dataclasses_json import Undefined, dataclass_json
from pydantic import BaseModel, Field
from typing_extensions import Literal

# TODO(yi): These docstrings are currently perfunctory. I'm not sure we even want to expose most of these
# fields. We need to overhaul our types :sadge:


class ModelBundleFrameworkType(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CUSTOM = "custom_base_image"


class PytorchFramework(BaseModel):
    framework_type: Literal[ModelBundleFrameworkType.PYTORCH]

    pytorch_image_tag: str
    """Image tag of the Pytorch image to use."""


class TensorflowFramework(BaseModel):
    framework_type: Literal[ModelBundleFrameworkType.TENSORFLOW]

    tensorflow_version: str
    """Tensorflow version to use."""


class CustomFramework(BaseModel):
    framework_type: Literal[ModelBundleFrameworkType.CUSTOM]

    image_repository: str
    """Docker image repository to use as the base image."""

    image_tag: str
    """Docker image tag to use as the base image."""


class ModelBundleFlavorType(str, Enum):
    CLOUDPICKLE_ARTIFACT = "cloudpickle_artifact"
    ZIP_ARTIFACT = "zip_artifact"
    RUNNABLE_IMAGE = "runnable_image"
    TRITON_ENHANCED_RUNNABLE_IMAGE = "triton_enhanced_runnable_image"


class CloudpickleArtifactFlavor(BaseModel):
    flavor: Literal[ModelBundleFlavorType.CLOUDPICKLE_ARTIFACT]

    requirements: List[str]
    """List of requirements to install in the environment before running the model."""

    framework: Union[PytorchFramework, TensorflowFramework, CustomFramework] = Field(
        ..., discriminator="framework_type"
    )
    """
    Machine Learning framework specification. Either
    [`PytorchFramework`](./#launch.model_bundle.PytorchFramework),
    [`TensorflowFramework`](./#launch.model_bundle.TensorflowFramework), or
    [`CustomFramework`](./#launch.model_bundle.CustomFramework).
    """

    app_config: Optional[Dict[str, Any]]
    """Optional configuration for the application."""

    location: str

    load_predict_fn: str
    """Function which, when called, returns the prediction function."""

    load_model_fn: str
    """Function which, when called, returns the model object."""


class ZipArtifactFlavor(BaseModel):
    flavor: Literal[ModelBundleFlavorType.ZIP_ARTIFACT]

    requirements: List[str]
    """List of requirements to install in the environment before running the model."""

    framework: Union[PytorchFramework, TensorflowFramework, CustomFramework] = Field(
        ..., discriminator="framework_type"
    )
    """
    Machine Learning framework specification. Either
    [`PytorchFramework`](./#launch.model_bundle.PytorchFramework),
    [`TensorflowFramework`](./#launch.model_bundle.TensorflowFramework), or
    [`CustomFramework`](./#launch.model_bundle.CustomFramework).
    """

    app_config: Optional[Dict[str, Any]]
    """Optional configuration for the application."""

    location: str

    load_predict_fn_module_path: str
    """Path to the module to load the prediction function."""

    load_model_fn_module_path: str
    """Path to the module to load the model object."""


class RunnableImageLike(BaseModel, ABC):
    """An abstract base for flavors that are related to bundles defined by runnable images."""

    repository: str
    tag: str
    command: List[str]
    env: Optional[Dict[str, str]]
    protocol: Literal["http"]  # TODO: add support for other protocols (e.g. grpc)
    readiness_initial_delay_seconds: int = 120


class RunnableImageFlavor(RunnableImageLike):
    """Model bundles that use custom docker images that expose an HTTP server for inference."""

    flavor: Literal[ModelBundleFlavorType.RUNNABLE_IMAGE]


class TritonEnhancedRunnableImageFlavor(RunnableImageLike):
    """For runnable image models that require tritonserver running in a container."""

    flavor: Literal[ModelBundleFlavorType.TRITON_ENHANCED_RUNNABLE_IMAGE]

    triton_model_repository: str

    triton_model_replicas: Optional[Dict[str, str]]

    triton_num_cpu: float

    triton_commit_tag: str

    triton_storage: Optional[str]

    triton_memory: Optional[str]

    triton_readiness_initial_delay_seconds: int = 300


ModelBundleFlavors = Union[
    CloudpickleArtifactFlavor,
    ZipArtifactFlavor,
    RunnableImageFlavor,
    TritonEnhancedRunnableImageFlavor,
]
"""Union type exhaustively representing all valid model bundle flavors.

Valid model bundle flavors are:
- [`CloudpickleArtifactFlavor`](./#launch.model_bundle.CloudpickleArtifactFlavor)
- [`ZipArtifactFlavor`](./#launch.model_bundle.ZipArtifactFlavor)
- [`RunnableImageFlavor`](./#launch.model_bundle.RunnableImageFlavor)
- [`TritonEnhancedRunnableImageFlavor`](./#launch.model_bundle.TritonEnhancedRunnableImageFlavor)
"""


class CreateModelBundleV2Response(BaseModel):
    """
    Response object for creating a Model Bundle.
    """

    model_bundle_id: str
    """ID of the Model Bundle."""


class ModelBundleV2Response(BaseModel):
    """
    Response object for a single Model Bundle.
    """

    id: str
    """ID of the Model Bundle."""

    name: str
    """Name of the Model Bundle."""

    metadata: Dict[str, Any]
    """Metadata associated with the Model Bundle."""

    created_at: datetime.datetime
    """Timestamp of when the Model Bundle was created."""

    model_artifact_ids: List[str]
    """IDs of the Model Artifacts associated with the Model Bundle."""

    schema_location: Optional[str]

    flavor: ModelBundleFlavors = Field(..., discriminator="flavor")
    """Flavor of the Model Bundle, representing how the model bundle was packaged.

    See [`ModelBundleFlavors`](./#launch_api.model_bundle.ModelBundleFlavors) for details.
    """


class ListModelBundlesV2Response(BaseModel):
    """
    Response object for listing Model Bundles.
    """

    model_bundles: List[ModelBundleV2Response]
    """A list of [Model Bundles](./#launch.model_bundle.ModelBundleV2Response)."""


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class ModelBundle:
    """
    Represents a ModelBundle.
    """

    name: str
    """
    The name of the bundle. Must be unique across all bundles that the user owns.
    """

    id: Optional[str] = None
    """
    A globally unique identifier for the bundle.
    """

    env_params: Optional[Dict[str, str]] = None
    """
    A dictionary that dictates environment information. See LaunchClient.create_model_bundle
    for more information.
    """

    location: Optional[str] = None
    """
    An opaque location for the bundle.
    """

    metadata: Optional[Dict[Any, Any]] = None
    """
    Arbitrary metadata for the bundle.
    """

    packaging_type: Optional[str] = None
    """
    The packaging type for the bundle. Can be ``cloudpickle`` or ``zip``.
    """

    requirements: Optional[List[str]] = None
    """
    A list of Python package requirements for the bundle. See LaunchClient.create_model_bundle
    for more information.
    """

    app_config: Optional[Dict[Any, Any]] = None
    """
    An optional user-specified configuration mapping for the bundle.
    """

    created_at: Optional[str] = None

    def __str__(self):
        return f"ModelBundle(bundle_name={self.name})"
