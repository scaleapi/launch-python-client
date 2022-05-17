from typing import Any, List, Sequence, Tuple, Type, Union

from launch_api.core import Service
from launch_api.types import I, O


class PipelineService(Service[I, O]):
    def __init__(
        self,
        service_classes_and_init_args: List[
            Union[
                Type[Service], Tuple[Type[Service], Union[Sequence[Any], Any]]
            ]
        ],
    ) -> None:
        self.services: List[Service] = []
        for x in service_classes_and_init_args:
            try:
                s_class, args = x
            except TypeError:
                s_class = x
                srv = s_class()
            else:
                try:
                    srv = s_class(*args)
                except TypeError:
                    srv = s_class(args)
            self.services.append(srv)

    def call(self, request: I) -> O:
        response = request
        for s in self.services:
            response = s.call(response)
        return response
