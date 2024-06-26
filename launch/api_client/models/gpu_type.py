# coding: utf-8

"""
    launch

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)

    The version of the OpenAPI document: 1.0.0
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


from __future__ import annotations

import json
from enum import Enum

from typing_extensions import Self


class GpuType(str, Enum):
    """
    Lists allowed GPU types for Launch.
    """

    """
    allowed enum values
    """
    NVIDIA_MINUS_TESLA_MINUS_T4 = "nvidia-tesla-t4"
    NVIDIA_MINUS_AMPERE_MINUS_A10 = "nvidia-ampere-a10"
    NVIDIA_MINUS_AMPERE_MINUS_A100 = "nvidia-ampere-a100"
    NVIDIA_MINUS_AMPERE_MINUS_A100E = "nvidia-ampere-a100e"
    NVIDIA_MINUS_HOPPER_MINUS_H100 = "nvidia-hopper-h100"
    NVIDIA_MINUS_HOPPER_MINUS_H100_MINUS_1G20GB = "nvidia-hopper-h100-1g20gb"
    NVIDIA_MINUS_HOPPER_MINUS_H100_MINUS_3G40GB = "nvidia-hopper-h100-3g40gb"

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create an instance of GpuType from a JSON string"""
        return cls(json.loads(json_str))
