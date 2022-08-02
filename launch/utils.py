from typing import Any, Dict, Optional


def trim_kwargs(kwargs_dict: Dict[Any, Any]):
    """
    Returns a copy of kwargs_dict with None values removed
    """
    dict_copy = {k: v for k, v in kwargs_dict.items() if v is not None}
    return dict_copy


def infer_env_params(env_selector: Optional[str]):
    """
    Returns an env_params dict from the env_selector.

    env_selector: str - Either "pytorch" or "tensorflow"
    """
    if env_selector == "pytorch":
        import torch

        try:
            ver = torch.__version__.split("+")
            torch_version = ver[0]
            cuda_version = ver[1][2:] if len(ver) > 1 else "113"
            if (
                len(cuda_version) < 3
            ):  # we can only parse cuda versions in the double digits
                raise ValueError(
                    "PyTorch version parsing does not support CUDA versions below 10.0"
                )
            tag = f"{torch_version}-cuda{cuda_version[:2]}.{cuda_version[2:]}-cudnn8-runtime"
            return {
                "framework_type": "pytorch",
                "pytorch_image_tag": tag,
            }
        except:
            raise ValueError(
                f"Failed to parse PyTorch version {torch.__version__}, try setting your own env_params."
            )
    elif env_selector == "tensorflow":
        import tensorflow as tf

        ver = tf.__version__
        return {
            "framework_type": "tensorflow",
            "tensorflow_version": ver,
        }
    else:
        raise ValueError(
            "Unsupported env_selector, please set to pytorch or tensorflow, or set your own env_params."
        )
