from enum import Enum
from typing import Any, Callable, Dict, Set, Type, Union

import pydantic
from pydantic import BaseModel

PYDANTIC_V2 = hasattr(pydantic, "VERSION") and pydantic.VERSION.startswith("2.")

if not PYDANTIC_V2:
    from pydantic.schema import (  # pylint: disable=no-name-in-module
        get_flat_models_from_models,
        model_process_schema,
    )


REF_PREFIX = "#/components/schemas/"


def get_model_definitions_v1(request_schema: Type[BaseModel], response_schema: Type[BaseModel]) -> Dict[str, Any]:
    """
    Gets the model schemas in jsonschema format from a sequence of Pydantic BaseModels.
    """
    flat_models = get_flat_models_from_models([request_schema, response_schema])
    model_name_map = {model: model.__name__ for model in flat_models}
    model_name_map.update({request_schema: "RequestSchema", response_schema: "ResponseSchema"})
    return get_model_definitions_from_flat_models(flat_models=flat_models, model_name_map=model_name_map)


def get_model_definitions_v2(request_schema: Type[BaseModel], response_schema: Type[BaseModel]) -> Dict[str, Any]:
    return {
        "RequestSchema": request_schema.model_json_schema(),  # type: ignore
        "ResponseSchema": response_schema.model_json_schema(),  # type: ignore
    }


if PYDANTIC_V2:
    get_model_definitions: Callable = get_model_definitions_v2  # type: ignore
else:
    get_model_definitions: Callable = get_model_definitions_v1  # type: ignore


def get_model_definitions_from_flat_models(
    *,
    flat_models: Set[Union[Type[BaseModel], Type[Enum]]],
    model_name_map: Dict[Union[Type[BaseModel], Type[Enum]], str],
) -> Dict[str, Any]:
    """
    Gets the model schemas in jsonschema format from a set of Pydantic BaseModels (or Enums).
    Inspired by https://github.com/tiangolo/fastapi/blob/99d8470a8e1cf76da8c5274e4e372630efc95736/fastapi/utils.py#L38

    Args:
        flat_models (Set[Union[Type[BaseModel], Type[Enum]]]): The models.
        model_name_map (Dict[Union[Type[BaseModel], Type[Enum]], str]): The map from model to name.

    Returns:
        Dict[str, Any]: OpenAPI-compatible schema of model definitions.
    """
    definitions: Dict[str, Dict[str, Any]] = {}
    for model in flat_models:
        m_schema, m_definitions, _ = model_process_schema(model, model_name_map=model_name_map, ref_prefix=REF_PREFIX)
        definitions.update(m_definitions)
        model_name = model_name_map[model]
        if "description" in m_schema:
            m_schema["description"] = m_schema["description"].split("\f")[0]
        definitions[model_name] = m_schema
    return definitions
