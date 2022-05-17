import logging
import subprocess
import tarfile
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple
from uuid import uuid4

from launch_api.core import Runtime, Runtimes, Service
from launch_api.model import NamedArrays, TritonModel

torch = Any
TfServingModel = Any
LaunchClient = Any

TritonClient = Any

logger = logging.getLogger("triton")

__all__: Sequence[str] = (
    "TritonModelServer",
    "make_pytorch_triton_configuration",
)


def download(model_repository_artifact: str) -> str:
    raise NotImplementedError


def extract(local_model_repository_archive: str) -> str:
    raise NotImplementedError


@dataclass
class TritonModelServer(Runtime):

    model_repository_artifact: str

    model_names_to_replication: Optional[Dict[str, int]]

    runtime_name = Runtimes.Triton

    def start(self) -> None:
        logger.info(
            f"Downloading & extracting model repository: {self.model_repository_artifact}"
        )
        dl = download(self.model_repository_artifact)
        local_model_repo = extract(dl)
        triton_cmd: List[str] = [
            "tritonserver",
            "--gpus",
            "all",
            "--model-repository",
            local_model_repo,
        ]
        if self.model_names_to_replication is not None:
            for m, i in self.model_names_to_replication.items():
                triton_cmd.extend(
                    [
                        f"--model={m}",
                        str(i),
                    ]
                )
            triton_cmd.extend(["--rate-limit", "execution_count"])

        logger.info("Starting Triton...")
        proc = subprocess.Popen(triton_cmd)
        proc.wait()


class ReferenceTritonModelServer(TritonModel):

    model_name: str
    triton_client: TritonClient
    use_grpc: bool = True

    def call(self, input_: NamedArrays) -> NamedArrays:
        result = self.triton_client.inference_async(
            name=self.model_name, input=input_
        )
        return result.get()


@dataclass(frozen=True)
class EnrichedTensor:
    data_type: str
    format: Optional[str]
    dims: List[int]


class ToProtobufText(ABC):
    @abstractmethod
    def as_pbtext(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class TritonTensor(EnrichedTensor, ToProtobufText):
    name: str
    data_type: str
    format: Optional[str]
    dims: List[int]

    def as_pbtext(self) -> str:
        s = "{\n"
        s += f'  name: "{self.name}"\n'
        s += f"  data_type: {self.data_type}\n"
        if self.format is not None:
            s += f"  format: {self.format}\n"
        s += f"  dims: {self.dims}\n"
        s += "}"
        return s.strip()


@dataclass(frozen=True)
class TritonModelConfig(ToProtobufText):
    model_name: str
    platform: str
    backend: Optional[str]
    max_batch_size: int
    dynamic_batching: Optional[float]
    input_shapes: List[TritonTensor]
    output_shapes: List[TritonTensor]

    def as_pbtext(self) -> str:
        s = ""
        s += f'name: "{self.model_name}"\n'
        s += f'platform: "{self.platform}"\n'
        if self.backend is not None:
            s += f'backend: "{self.backend}"\n'
        s += f"max_batch_size : {self.max_batch_size}\n"
        if self.dynamic_batching is not None:
            s += "dynamic_batching {\n"
            s += f"  max_queue_delay_microseconds: {self.dynamic_batching}\n"
            s += "}\n"
        _inputs = [i.as_pbtext() for i in self.input_shapes]
        s += f"input: {_inputs}\n"
        _outptus = [o.as_pbtext() for o in self.output_shapes]
        s += f"output: {_outptus}\n"
        return s.strip()


class RunnableTritonConfig(NamedTuple):
    model_artifact_uri: str
    config: TritonModelConfig


def make_pytorch_triton_configuration(
    models_to_configure_for_triton: List[
        Tuple[torch.Module, TritonModelConfig]
    ],
    s3_client: Any,
    gzip_compression: bool = False,
    backend_for_model_artifact_storage: str = "s3://scale-ml/home/malcolmgreaves/prod/model_artifact_base",
) -> Dict[str, RunnableTritonConfig]:

    # validate
    for model, triton_model_configuration in models_to_configure_for_triton:
        assert triton_model_configuration.platform == "pytorch_libtorch"
        assert isinstance(model, torch.Module)

    # torchscript models
    jitted_models_to_configurations = []
    for model, triton_model_configuration in models_to_configure_for_triton:
        try:
            jitted_model = torch.jit.script(model)
        except:
            logger.exception(
                f"Failed to TorchScript model ({triton_model_configuration.model_name})"
            )
            raise
        else:
            jitted_models_to_configurations.append(
                (jitted_model, triton_model_configuration)
            )

    saved_artifacts: Dict[str, RunnableTritonConfig] = dict()
    # create model artifact file structure
    # save relevant data
    # archive it
    # then upload
    with tempfile.TemporaryDirectory() as _tempdir:
        tempdir = Path(_tempdir)

        for i, (jitted_model, triton_model_configuration) in enumerate(
            jitted_models_to_configurations
        ):
            # NOTE: Must start at 1 for Triton!
            i += 1

            # make model artifact directory file & directory structure
            artifact_dir: Path = (
                tempdir / triton_model_configuration.model_name
            )
            artifact_dir.mkdir()

            model_dir: Path = artifact_dir / f"{i}"
            model_dir.mkdir()

            # save the model
            model_fi: Path = model_dir / "model.pt"
            jitted_model.save(str(model_fi.absolute()))

            # save the Triton configuration for the model
            config_fi: Path = artifact_dir / "config.pbtxt"
            with config_fi.open("wt") as wt:
                wt.write(triton_model_configuration.as_pbtext())

            # create an archive, optionally compressing it with gzip
            tar_name = f"{triton_model_configuration.model_name}.tar"
            if gzip_compression:
                tar_name += ".gz"
                mode = "w:gz"
            else:
                mode = "w"
            with tarfile.open(tar_name, mode) as w:
                w.add(str(artifact_dir), recursive=True, filter=None)

            # upload the archive
            s3_location = (
                f"{backend_for_model_artifact_storage}/{uuid4()}/{tar_name}"
            )
            s3_client.copy(tar_name, s3_location)
            logger.info(
                f"[{i}/{len(jitted_models_to_configurations)}] Formatted and saved model "
                f'"{triton_model_configuration.model_name}" to {s3_location}'
            )

            saved_artifacts[
                triton_model_configuration.model_name
            ] = RunnableTritonConfig(
                model_artifact_uri=s3_location,
                config=triton_model_configuration,
            )

        return saved_artifacts


def triton_config_pytorch_model(
    *,
    model_name: str,
    input_shapes: List[EnrichedTensor],
    output_shapes: List[EnrichedTensor],
    max_batch_size: int,
    dynamic_batching: Optional[float] = None,
) -> TritonModelConfig:
    def convert(
        shapes: List[EnrichedTensor], *, is_input: bool
    ) -> List[TritonTensor]:
        prefix = "input" if is_input else "output"
        return [
            TritonTensor(
                name=f"{prefix}__{i}",
                data_type=shape.data_type,
                format=shape.format,
                dims=shape.dims,
            )
            for i, shape in enumerate(shapes)
        ]

    triton_input_shapes = convert(input_shapes, is_input=True)
    triton_output_shapes = convert(output_shapes, is_input=False)

    return TritonModelConfig(
        model_name=model_name,
        input_shapes=triton_input_shapes,
        output_shapes=triton_output_shapes,
        max_batch_size=max_batch_size,
        dynamic_batching=dynamic_batching,
        platform="pytorch_libtorch",
        backend=None,
    )


def triton_config_tensorflow_model(
    *,
    model_name: str,
    decorated_tensorflow_model: TfServingModel,
    max_batch_size: int,
    dynamic_batching: Optional[float] = None,
) -> TritonModelConfig:

    return TritonModelConfig(
        model_name=model_name,
        input_shapes=decorated_tensorflow_model.input_shapes,
        output_shapes=decorated_tensorflow_model.input_shapes,
        max_batch_size=max_batch_size,
        dynamic_batching=dynamic_batching,
        platform="tensorflow_savedmodel",
        backend="tensorflow",
    )


def demo():
    @dataclass
    class MyModel(TritonModel):
        model: torch.Module

        def call(self, array):
            assert len(array) == 4
            assert array.shape[0] >= 1
            assert array.shape[1:] == (255, 255, 3)
            preds = self.model(array)
            assert preds.shape[0] == array.shape[0]
            assert preds.shape[1:] == (1000,)
            return preds

    service = MyModel(...)

    triton_format_model = triton_config_pytorch_model(
        model_name="mymodel",
        input_shapes=[
            EnrichedTensor(
                data_type="TYPE_INT32", format=None, dims=[255, 255, 3]
            )
        ],
        output_shapes=[
            EnrichedTensor(data_type="TYPE_FLOAT16", format=None, dims=[1000])
        ],
        max_batch_size=8,
        dynamic_batching=50000,
    )

    runnable_configs = make_pytorch_triton_configuration(
        [(service.model, triton_format_model)],
        s3_client=None,
    )
    assert len(runnable_configs.items()) == 1
    model_name, runnable_config = tuple(runnable_configs.items())
    assert model_name == "mymodel"
    assert isinstance(runnable_config, RunnableTritonConfig)

    logger.info(
        f"The following model artifact: {runnable_config.model_artifact_uri}"
    )
    logger.info(
        f"Will work with the following Triton model configuration:\n{runnable_config.config.as_pbtext()}"
    )

    runnable = TritonModelServer(
        model_repository_artifact=runnable_config.model_artifact_uri,
        model_names_to_replication={runnable_config.config.model_name: 1},
    )
    logger.info("Now running...")
    runnable.start()

    client: LaunchClient = None

    reference: ReferenceTritonModelServer = client.triton_standin(
        triton_format_model
    )

    JsonVal = Any
    import numpy as np

    @dataclass
    class MyServiceThatUsesTriton(Service):

        preprocess: Service[JsonVal, np.ndarray]
        triton: Service[NamedArrays, NamedArrays]
        postprocess: Service[np.ndarray, JsonVal]

        def call(self, request: JsonVal) -> JsonVal:
            model_input = self.preprocess(request)
            model_output = self.triton({"input__0": model_input})
            response = self.postprocess(model_output["output__0"])
            return response

    from launch_api.loader import ServiceLoader

    Preprocess = Postprocess = lambda: None

    class LoadMyServiceThatUsesTriton(ServiceLoader[MyServiceThatUsesTriton]):

        preprocess_args: dict
        triton_format_model: TritonModelConfig
        postprocess_args: dict

        def load(self) -> MyServiceThatUsesTriton:
            return MyServiceThatUsesTriton(
                Preprocess(),
                client.find_triton_matching(triton_format_model),
                Postprocess(),
            )

    class ServiceDescription:
        pass

    class Deployment:
        service: ServiceDescription
        trtion: Optional[TritonModelConfig]


# triton_format_model = triton_helpers.convert_pytorch_model(
# 	service.model,
# 	input_shapes=[(255, 255, 3)],
# 	output_shapes[(1000,)],
# )
# # either OK or failure
# if isinstance(triton_format_model, Error):
# 	raise ValueError(triton_format_model)
#
# triton_format_model.data  # what to save on disk!
# triton_format_model.model  # torch scripted model!
# triton_format_model.input_shapes  # {"input__0": (255, 255, 3)}
# triton_format_model.output_shapes  #
#
# saved = triton_format_model.save('./model.pt')
# model_registry.triton.name[model_artifact_uri] = saved
# model_registry.triton.upload()
#
# # now on image:
# here = model_registry.triton.name[model_artifact_uri].download()
# TritonModelServer(here).start()
