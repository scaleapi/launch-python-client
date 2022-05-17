from dataclasses import dataclass
from typing import Optional, Union

from flask import Flask, Response, jsonify, request
from launch_api.core import Runtime, Service
from launch_api.types import I, O


@dataclass
class HttpRuntime(Runtime):

    service: Service[I, O]
    port: int
    name: Optional[str]
    is_debug: bool = False

    def start(self) -> None:
        app = HttpRuntime.server(
            self.service,
            self.name if self.name is not None else "http-service",
        )
        app.run(port=self.port, debug=self.is_debug)

    @staticmethod
    def server(
        service: Service[I, O], existing_app_or_name: Union[Flask, str]
    ) -> Flask:

        if isinstance(existing_app_or_name, str):
            app: Flask = Flask(existing_app_or_name)
        else:
            app = existing_app_or_name

        @app.route("/healthcheck", methods=["GET"])
        def healthcheck():
            return Response(status=200, headers={})

        @app.route("/predict", methods=["POST"])
        def predict():
            payload = request.get_json()
            response = service.call(payload)
            return jsonify(response)

        return app
