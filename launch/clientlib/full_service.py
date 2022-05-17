import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

from launch_api.core import Service
from launch_api.types import I, O

logger = logging.getLogger("full_service")


class RequestHandler(Generic[I], ABC):
    @abstractmethod
    def deserialize(self, request: bytes) -> I:
        raise NotImplementedError()


class ResponseHandler(Generic[O], ABC):
    @abstractmethod
    def serialize(self, response: O) -> bytes:
        raise NotImplementedError()


@dataclass
class RunnableService(Generic[I, O]):
    request_handler: RequestHandler[I]
    service: Service[I, O]
    response_handler: ResponseHandler[O]

    def call(self, request: bytes) -> bytes:
        # deserialize request payload into format service can handle
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
            response: bytes = self.response_handler.serialize(output)
        except:
            logger.exception(
                f"Could not serialize service output ({type(output)=} | {output=})"
            )
            raise
        # serialized response for sending over network
        return response
