from typing import List, Optional

from pydantic import BaseModel

from launch.pydantic_schemas import (
    get_model_definitions,
    get_model_definitions_from_flat_models,
)


def test_get_model_definitions():
    class MyRequestSubSchemaB(BaseModel):
        query: str
        language: str

    class MyRequestSchemaB(BaseModel):
        queries_and_languages: List[MyRequestSubSchemaB]
        temperature: Optional[float]

    class MyResponseSchemaB(BaseModel):
        responses: List[str]
        total_num_tokens: int
        time_elapsed: float

    result = get_model_definitions(
        request_schema=MyRequestSchemaB, response_schema=MyResponseSchemaB
    )

    expected = {
        "MyRequestSubSchemaB": {
            "title": "MyRequestSubSchemaB",
            "type": "object",
            "properties": {
                "query": {"title": "Query", "type": "string"},
                "language": {"title": "Language", "type": "string"},
            },
            "required": ["query", "language"],
        },
        "RequestSchema": {
            "title": "MyRequestSchemaB",
            "type": "object",
            "properties": {
                "queries_and_languages": {
                    "title": "Queries And Languages",
                    "type": "array",
                    "items": {
                        "$ref": "#/components/schemas/MyRequestSubSchemaB"
                    },
                },
                "temperature": {"title": "Temperature", "type": "number"},
            },
            "required": ["queries_and_languages"],
        },
        "ResponseSchema": {
            "title": "MyResponseSchemaB",
            "type": "object",
            "properties": {
                "responses": {
                    "title": "Responses",
                    "type": "array",
                    "items": {"type": "string"},
                },
                "total_num_tokens": {
                    "title": "Total Num Tokens",
                    "type": "integer",
                },
                "time_elapsed": {"title": "Time Elapsed", "type": "number"},
            },
            "required": ["responses", "total_num_tokens", "time_elapsed"],
        },
    }

    assert result == expected


def test_get_model_definitions_from_flat_models():
    class MyRequestSchema(BaseModel):
        x: int
        y: str

    class MyResponseSchema(BaseModel):
        __root__: int

    flat_models = {MyRequestSchema, MyResponseSchema}
    model_name_map = {
        MyRequestSchema: "RequestSchema",
        MyResponseSchema: "ResponseSchema",
    }

    result = get_model_definitions_from_flat_models(
        flat_models=flat_models, model_name_map=model_name_map
    )
    expected = {
        "RequestSchema": {
            "title": "MyRequestSchema",
            "type": "object",
            "properties": {
                "x": {"title": "X", "type": "integer"},
                "y": {"title": "Y", "type": "string"},
            },
            "required": ["x", "y"],
        },
        "ResponseSchema": {"title": "MyResponseSchema", "type": "integer"},
    }

    assert result == expected
