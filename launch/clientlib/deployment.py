from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import List


class DeployedService(ABC):
    @abstractmethod
    def send(self, request) -> Future:
        raise NotImplementedError


# class Deployment(DeployedService):
#     services_in_order: List[DeployedService]
#
#     def call(self, request) -> Future:
#         x = request
#         for f in self.services_in_order:
#             pending: Future = f.send(x)
#             x = pending.result()
#
#         result = Future()
#         result.set_result(x)
#         return result


# deployment onto a pod is either:
# - 1 container: just the Service()
# - 2 containers: the Service() and then a tritonserver()
# NO other multi-container situations on a pod!!
