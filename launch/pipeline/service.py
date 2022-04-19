import inspect
from typing import Any, Callable, Dict, List, Optional

from launch.pipeline.deployment import Deployment
from launch.pipeline.runtime import Runtime


class ServiceDescription:
    """
    Base ServiceDescription.
    """

    def call(self, req: Any) -> Any:
        raise NotImplementedError()


class SingleServiceDescription(ServiceDescription):
    """
    The description of a service.
    """

    def __init__(
        self,
        service: Callable,
        runtime: Runtime,
        deployment: Deployment,
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.service = service
        self.runtime = runtime
        self.deployment = deployment
        self.kwargs = kwargs

        self._callable_service: Optional[Callable] = None

    def _init_callable_service(self):
        if self._callable_service:
            return
        if inspect.isclass(self.service):
            self._callable_service = self.service(**self.kwargs)
        else:
            assert (
                not self.kwargs
            ), "`kwargs` is given, but the service is not a class"
            self._callable_service = self.service

    def call(self, req: Any) -> Any:
        self._init_callable_service()
        assert self._callable_service, "`_callable_service` is not initialized"
        return self._callable_service(req)


class SequentialPipelineDescription(ServiceDescription):
    """
    The description of a sequential pipeline.
    """

    def __init__(
        self,
        items: List[SingleServiceDescription],
    ):
        super().__init__()
        self.items = items

    def call(self, req: Any) -> Any:
        res = req
        for item in self.items:
            res = item.call(res)
        return res
