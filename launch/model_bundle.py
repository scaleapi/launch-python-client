from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dataclasses_json import Undefined, dataclass_json

# TODO(yi): These docstrings are currently perfunctory. I'm not sure we even want to expose most of these
# fields. We need to overhaul our types :sadge:


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

    bundle_id: Optional[str] = None
    """
    A globally unique identifier for the bundle. This is not to be used in the API.
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

    def __str__(self):
        return f"ModelBundle(bundle_name={self.name})"
