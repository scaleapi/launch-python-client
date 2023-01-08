from pydantic import BaseModel

from launch.pydantic_schemas import get_model_definitions


def test_get_model_definitions():
    class MyRequestSchema(BaseModel):
        x: int
        y: str

    class MyResponseSchema(BaseModel):
        __root__: int

    flat_models = {MyRequestSchema, MyResponseSchema}
    model_name_map = {MyRequestSchema: "RequestSchema", MyResponseSchema: "ResponseSchema"}

    result = get_model_definitions(flat_models=flat_models, model_name_map=model_name_map)
    expected = {
        "RequestSchema": {
            "title": "MyRequestSchema",
            "type": "object",
            "properties": {
                "x": {
                    "title": "X",
                    "type": "integer"
                },
                "y": {
                    "title": "Y",
                    "type": "string"
                }
            },
            "required": [
                "x",
                "y"
            ]
        },
        "ResponseSchema": {
            "title": "MyResponseSchema",
            "type": "integer"
        }
    }

    assert result == expected
