import inspect
import uuid
from typing import Any, Callable, Dict, List, Optional

from launch.pipeline.deployment import Deployment
from launch.pipeline.runtime import Runtime

MAX_SERVICE_NAME_SIZE = 50


class ServiceDescription:
    """
    Base ServiceDescription.
    """

    service_name: str
    service: Callable
    runtime: Runtime
    deployment: Deployment
    env_params: Dict[str, str]

    def __init__(self):
        self.make_request: Optional[Callable] = None

    def set_make_request(
        self, make_request: Callable, force: bool = False
    ) -> None:
        raise NotImplementedError()

    def call(self, req: Any) -> Any:
        raise NotImplementedError()

    def __call__(self, req: Any) -> Any:
        return self.call(req)


class SingleServiceDescription(ServiceDescription):
    """
    The description of a service.
    """

    def __init__(
        self,
        service: Callable,
        runtime: Runtime,
        deployment: Deployment,
        env_params: Dict[str, str],
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        setattr(self, "service", service)
        self.runtime = runtime
        self.deployment = deployment
        self.env_params = env_params
        self.kwargs = kwargs
        func_name = service.__name__.replace("_", "").replace("-", "")
        self.service_name = f"{func_name}{uuid.uuid4()}"[
            :MAX_SERVICE_NAME_SIZE
        ]

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

    def set_make_request(
        self, make_request: Callable, force: bool = False
    ) -> None:
        if force:
            self.make_request = make_request

    def call(self, req: Any) -> Any:
        if self.make_request:
            return self.make_request(
                servable_id=self.service_name,
                local_fn=self.service,
                args=[req],
                kwargs={},
            )
        else:
            self._init_callable_service()
            assert (
                self._callable_service
            ), "`_callable_service` is not initialized"
            return self._callable_service(req)


class SequentialPipelineDescription(ServiceDescription):
    """
    The description of a sequential pipeline.
    """

    def __init__(
        self,
        items: List[SingleServiceDescription],
        runtime: Runtime,
        deployment: Deployment,
        env_params: Dict[str, str],
    ):
        super().__init__()
        setattr(
            self, "service", self
        )  # Pipeline is itself a service via the __call__ function
        self.items = items
        self.runtime = runtime
        self.deployment = deployment
        self.env_params = env_params
        self.service_name = f"pipeline{uuid.uuid4()}"[:MAX_SERVICE_NAME_SIZE]

    def set_make_request(
        self, make_request: Callable, force: bool = False
    ) -> None:
        if force:
            self.make_request = make_request
        for item in self.items:
            item.set_make_request(make_request, force=True)

    def call(self, req: Any) -> Any:
        res = req
        for item in self.items:
            res = item.call(res)
        return res
