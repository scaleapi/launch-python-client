from enum import Enum
from typing import Union, Type, Set, Dict, Any

from pydantic.schema import model_process_schema
from pydantic import BaseModel

REF_PREFIX = "#/components/schemas/"


def get_model_definitions(
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
        m_schema, m_definitions, m_nested_models = model_process_schema(
            model, model_name_map=model_name_map, ref_prefix=REF_PREFIX
        )
        definitions.update(m_definitions)
        model_name = model_name_map[model]
        if "description" in m_schema:
            m_schema["description"] = m_schema["description"].split("\f")[0]
        definitions[model_name] = m_schema
    return definitions
