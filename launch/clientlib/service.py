import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from launch_api.core import Service
from launch_api.types import I, JsonVal, O

logger = logging.getLogger("service")

__all__: Sequence[str] = (
    # re-export
    "Service",
    # data type generic parameter
    "D",
    # service sub-structure
    "RequestHandler",
    "ResponseHandler",
    "FullService",
)

D = TypeVar("D")


class RequestHandler(Generic[D, I], ABC):
    """Responsible for converting protocol-formatted data (D) into a service's input type (I)."""

    @abstractmethod
    def deserialize(self, request: D) -> I:
        raise NotImplementedError()


class ResponseHandler(Generic[D, O], ABC):
    """Responsible for converting a service's output (O) into the protocol's data format (D)."""

    @abstractmethod
    def serialize(self, response: O) -> D:
        raise NotImplementedError()


@dataclass
class FullService(Generic[D, I, O], Service[D, D]):
    """The thing that we technically run: service + payload serializers.

    Its the service logic + knowing how to encode & decode things into the transit data format.
    Default implementation is to use JSON formatted strings.
    """

    service: Service[I, O]

    request_handler: RequestHandler[D, I]

    response_handler: ResponseHandler[D, O]

    # cannot be overridden!
    def call(self, request: D) -> D:
        # deserialize JSON into format service can handle
        try:
            input_: I = self.request_handler.deserialize(request)
        except:
            logger.exception(
                f"Could not deserialize request ({type(request)=} | {request=})"
            )
            raise
        # run service logic
        try:
            output: O = self.service.call(input_)
        except:
            logger.exception(
                f"Could not perform service calculation ({type(input_)=})"
            )
            raise
        # serialize service output into JSON
        try:
            response: JsonVal = self.response_handler.serialize(output)
        except:
            logger.exception(
                f"Could not serialize service output ({type(output)=} | {output=})"
            )
            raise
        # callers get JSON response
        return response


class JsonService(FullService[str, JsonVal, JsonVal]):
    """This is what all std-ml-srv services are, effectively.

    + all services accept and return JSON-formatable values
    + all protocols encode the data a JSON-formatted strings
    """

    def __init__(self, service: Service[I, O]) -> None:
        super().__init__(
            service=service,
            request_handler=_J,
            response_handler=_J,
        )


class JsonHandler(RequestHandler[str, JsonVal], ResponseHandler[str, JsonVal]):
    def serialize(self, response: JsonVal) -> str:
        return json.dumps(response)

    def deserialize(self, request: str) -> JsonVal:
        return json.loads(request)


_J = JsonHandler()


def default_full_service(s: Service) -> FullService:
    return JsonService(s)
