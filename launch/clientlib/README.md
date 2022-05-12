# Launch API
**Author**: Malcolm Greaves \
**Last Major Modification Date**: Wednesday, April 20, 2020

The Launch API is centered around three major abstractions that intersect to define a fully realized machine learning
inference service.  These abstractions define what is run, how it is run, and where the execution occurs. Respectively,
these concepts  are referred to as a `Service` (the "what"), a `Runtime` (the "how"), and a `Deployment` (the "where").

The rest of this document will describe these major abstractive components in detail as well as showcase their intended use.
Moreover, this document will show how users can evolve a ML inference service to increase their code's modularity with
mild increases in complexity. This modularity-complexity tradeoff ramp grows to include computational benefits as well,
showing how users can accelerate the tensor-based operations of their model inference using Triton.

The major sections of this document are:
- [Core Concepts](#core-concepts)
- [Using the API](#using-the-api)
- [Batteries Included Services & Batching](#batching-and-batteries-included-services)
- [Triton](#triton)
- [Handling Different Serialized Request Formats](#handling-different-serialized-request-formats)

## Core Concepts

Defining the "what", the "how", and the "where" of a complete ML inference service.

### Service
A `Service` in Launch is, at its core, a user-defined function. It is completely customizable logic that the user writes.
This `Service` logic is responsible for performing model inference as the users sees fit. It accepts the request as its 
only input and produces the inference service's response as its only output.
[The complete definition is here](./launch_api/core.py), but a simplified version is reproduced below:
```python
class Service:
    def call(self, request: I) -> O:
        ...
```

Crucially, a `Service` is stateless: after initialization, there can be no state mutation that is preserved between 
successive invocations of its `call` method. This stateless property is important to ensure correctness in horizontal 
scaling.

A core design principle of Launch is that a user must define what their logic is within this `call` method.
Once complete, the rest of the Launch framework is able to handle the running and deployment of a real synchronous or
asynchronous service with their custom-defined logic.

### Runtime
A `Runtime` is a user-selectable backend that is able to provide networked access to the user's `Service` logic. A
`Runtime` is responsible for startup, listening for requests over a network, and coordinating handoff of those requests 
to the user's `Service` logic. It is also responsible for ensuring that exceptions raised in the user's `Service` 
while handling a request do not propagate outward and cause the `Runtime` to fail to serve additional requests.

Even simplified, [the `Runtime`'s API is quite limited](./launch_api/core.py), as its sole responsibility is being able 
to start handling requests: 
```python
class Runtime:
    def start(self) -> None:
        ...
```
Initial `Runtime` implementations will be HTTP, Celery, and Triton-based backends.
An additional goal for Launch is to allow users to define custom backends that can inter-operate with the Launch
framework. For example, while Launch will not support GRPC based ML inference services, it should be possible for an 
advanced user to define a compatible `Runtime` and plug it into the Launch system.

### Deployment
A `Deployment` is the pairing of a `Service` and `Runtime`, along with `DeploymentOptions`, to create a 
production-grade ML inference service. By themselves a `Runtime` serving the logic of a user's `Service` is enough to
provide actual inference serving capability. However, this still leaves as gap: _**where**_ is the execution occurring?

For demonstration and development, a workstation or laptop might be a suitable location for serving inference requests.
However, in production, it is important to have ML inference services executing in such a way as to adapt to varying
request loads, not have a single point of failure, and have tooling that makes it easy to redeploy, move deployments
between different computing environments, and rollback bad or otherwise broken deployments. Thus, it is crucial to
enable users to deploy their services onto a cluster of machines in a manner that provides for their production needs 
without requiring users to understand the intricacies of building a robust distributed system.

A `Deployment` can mean several things: it can be the actual set of compute nodes currently running a user's `Service`
logic or it can be an actionable description thereof. In the former case, we define a `DeployedService` that acts
as an opaque reference to some actual running `Deployment`. This `DeployedService` allows a user to inspect the status 
of their deployed service as well as send requests to it. In contrast, a `Deployment` is specifically the compute plan
for a `Service` and `Runtime`. It specifies details such as autoscaling behavior, hardware requirements, tracking 
metadata, etc. In turn, the Launch system knows how to perform the actual deployment actions that are specified within 
a `Deployment` instance. Launch can turn a `Deployment` into a `DeployedService`.

Below is a simplified API of a `Deployment`:
```python
class Deployment:
    def plan(self) -> str:
        """K8s deployment file contents."""

    def service_description(self) -> str:
        """Description of the service(s) to be run."""

    def runtime_description(self) -> str:
        """Description of the runtime(s) that the service(s) are being run on."""

    def runnable_service_bundle(self) -> str:
        """Unique identifier for the required environment bundle."""

    def options(self) -> DeploymentOptions:
        """The customizable deployment options, including specifications for autoscaling behavior, hardware, etc."""
```

In contrast, a `DeployedService` has a smaller surface area:
```python
Status = "not started" | "pending" | "starting" | "running" | "error" 

class DeployedService:
    
    def send(self, request: I) -> Future[O]:
        ...

    def status(self) -> Status:
       ...
```

While not complete, `DeploymentOptions` will look something like:
```python
@dataclass(frozen=True)
class DeploymentOptions:
    min_workers: int
    max_workers: int
    target_pod_deployment: float
    cpu: float
    mem: float
    gpus: Optional[int]
    machine_type: Optional[str]
    ...
```


## Using the API

### Using the Launch Client to Deploy
Given the Launch client (stand-in type `LaunchClient`), a user would do the following to perform a deployment of their
service as:
```python
client: LaunchClient = ...

# A user must create a portable environment with their Python code,
# and all other dependencies, so that Launch is able to run their code
# in a remote environment.
making_bundle = client.create_bundle(
    # the service class
    service=...,
    # the initialization arguments for loading the service
    initialization=...,
    # the user's dependencies
    environment=...,
)
# bundle creation occurs remotely
# if the docker image fails to build, this call will result in an exception
bundle = making_bundle.get()

deploy_opts: DeploymentOptions = ...
# these deployment options are completely user specified

deployment: Deployment = client.create_deployment(
    service_bundle=bundle,
    # There will be a set of standard runtimes for Launch
    # But a user could specify their own class here.
    runtime=Runtimes.SNYC,
    deployment_options=deploy_opts,
)

ref: DeployedService = client.deploy(deployment)
# actual k8s deployment is now occurring, this usually takes several minutes
response = ref.send({"request":"payload"})
# blocks until the deployment has started and the runtime has started accepting requests
#  - if the deployment fails, then the Future will resolve to an Exception
#  - if it is successful, then this will eventually be a valid inference result
print(response.get())
```

### Basic Use: Writing a `Service` Quickly
Let's say that you want to deploy a binary classification model service. This particular trained model knows how to 
differentiate between pictures that contain hot dogs and those that do not. Given some `image_classifier` PyTorch model,
we will demonstrate the `Service` code that a user would need to implement:
```python
class HotdogOrNot(Service[Dict[str, str], Dict[str, str]]):
    
    image_classifier: torch.Module
    device: torch.device

    def __init__(self, model_path: str) -> None:
        self.device = torch.device("GPU" if torch.cuda.is_available() else 'cpu')
        self.image_classifier = torch.load(model_path, map_location=self.device) 

    def call(self, request: Dict[str, str]) -> Dict[str, str]:
        image_url = request['url']
        data: bytes = download(image_url)
        img: PIL.Image.Image = PIL.Image.Image.open(data, convert_rgb=True)
        
        model_input = torch.tensor(np.asarray(img)).to(self.device)
        model_output: torch.Tensor = self.image_classifier(model_input)
        preds: np.ndarray = model_output.detach().cpu().array()
        
        confidence = preds[0]
        prediction = "hotdog" if confidence > 0.5 else "not hotdog"
        return {'prediction': prediction, 'confidence': confidence}
```

The `HotdogOrNot` service is initialized by downloading a PyTorch model artifact. The service's `call` method parses 
a JSON-like payload to extract an image's URL. This image is then downloaded and formatted into a dense n-dimensional 
array for the model. The model then performs inference and its results are interpreted for a final classification
decision. This result is packed into another JSON-like object and is returned.

Given the Launch API, a user would be able to go from this code and a few dependencies (`pytorch`, `PIL`, `numpy`) and
be able to construct a complete working production-grade, highly-available, horizontally scalable service.

### Intermediate Use: Modularity and Code Reuse

As a codebase grows, it is extremely likely for a user to require the same functionality in multiple places. Users 
organically refactor and re-use code as the need arises. Likewise, Launch must be able to grow with a user as their
business needs and complexity of implementation grows.

Taking the previous section's example, we will assume that a user now has multiple services for doing image
classification. Thus, they will need generalized image downloading and preprocessing as a reusable component. Likewise,
they will reuse the same model loading and inference code. This is how a user would refactor their code to make these 
two reusable components:
```python
class ImageDownloader(Service[Dict[str,str], np.ndarray]):
    def call(self, request: Dict[str,str]) -> np.ndarray:
        image_url = request['url']
        data: bytes = download(image_url)
        img: PIL.Image.Image = PIL.Image.Image.open(data, convert_rgb=True)
        return np.asarray(img)


class PytorchModel(Service[np.ndarray, np.ndarray]):
    
    image_classifier: torch.Module
    device: torch.device
    
    def __init__(self, model_path: str) -> None:
        self.device = torch.device("GPU" if torch.cuda.is_available() else 'cpu')
        self.image_classifier = torch.load(model_path, map_location=self.device) 

    def call(self, request: np.ndarray) -> np.ndarray:
        model_input = torch.tensor(request).to(self.device)
        model_output: torch.Tensor = self.image_classifier(model_input)
        preds: np.ndarray = model_output.detach().cpu().array()
        return preds
```

Their codebase would have different components that would accept the output of the `PytorchModel` service and
interpret the predictions in ways that are aligned with the business that they're supporting. For example, their
original hotdog detector would become:
```python
class HotdogOrNotPostproccessor(Service[np.ndarray, Dict[str,str]]):
    def call(self, request: np.ndarray) -> Dict[str,str]:
        confidence = request[0]
        prediction = "hotdog" if confidence > 0.5 else "not hotdog"
        return {'prediction': prediction, 'confidence': confidence}
```

Now, the user can regain their original `HotDogService` functionality by combining their smaller, individual services:
```python
class HotDogServiceV2(Service[Dict[str,str], Dict[str,str]):
    def __init__(self, model_path: str) -> None:
        self.preprocess = ImageDownloader()
        self.postprocess = HotDogOrNotPostprocessor()
        self.model = PytorchModel(model_path)

    def call(self, request: Dict[str,str]) -> Dict[str,str]:
        image = self.preprocess.call(request)
        preds = self.model.call(image)
        return self.postprocess.call(preds)
```

 
If instead the `image_classifier` was trained on a multi-class dataset, such as ImageNet, a user could gain this
post-processing functionality by creating:
```python
class ImageNetPostprocessor(Service[np.ndarray, Dict[str,str]]):
    
    imagenet_classes: Sequence[str]
    
    def __init__(self, imagenet_classes_url: str) -> None:
        self.imagenet_classes = json.loads(download(imagenet_classes_url))
        
    def call(self, request: np.ndarray) -> Dict[str,str]:
        return {
            'prediction': self.imagenet_classes[prediction_class_index], 
            'confidence': request[prediction_class_index],
        }
```
And using it as the new `postprocess` step for an entirely different `Service` class:  
```python
class ImageNetService(Service[Dict[str,str], Dict[str,str]]):
    
    def __init__(self, model_path: str, imagenet_classes_url: str) -> None:
        self.preprocess = ImageDownloader()
        self.model = PytorchModel(model_path)
        self.postprocess = ImageNetPostprocessor(imagenet_classes_url)

    def call(self, request: Dict[str,str]) -> Dict[str,str]:
        image = self.preprocess.call(request)
        preds = self.model.call(image)
        return self.postprocess.call(preds)
```

### Advanced Use

#### Pipelines
The previous example showed repetition: as the user generalized their service to reuse code for other applications, the
meta computation of their inference application emerged as a pipeline of preprocessing, model inference, and postprocessing.
Pipelining is a normal and common abstraction for data transformations.

While users are always free to define an arbitrarily complex data transformation flow between any number of existing
services, the Launch API should provide a mechanism to allow users to easily create such a pipelined service.
A [`PipelineService` illustrating initialization and the `call` method is included here](./launch_api/pipeline.py).
It should accept a list, in execution order, of `Service` classes as well as their initialization arguments, if applicable.

Users can use by directly extending the `PipelineService` class into their own:
```python
class ImagenetService(PipelineService[Dict[str,str], Dict[str,str]]):
    def __init__(self, model_path: str, imagenet_classes_url: str) -> None:
        super().__init__(
            [
                ImageDownloader,
                (PytorchModel, model_path),
                (ImageNetPostprocessor, imagenet_classes_url),
            ]
        )
```
Or, they can use the `LaunchClient` to make such a service dynamically:
```python
imagenet_service: Service[Dict[str,str], Dict[str,str]] = client.service.pipeline(
    [
        ImageDownloader,
        (PytorchModel, model_path),
        (ImageNetPostprocessor, imagenet_classes_url),
    ]
)

imagenet_service.call({"url": "..."})
```

And deploy just as easily: 
```python
making_bundle = client.create_bundle(
    service=PipelineService,
    initialization_arguments=[
        ImageDownloader,
        (PytorchModel, model_path),
        (ImageNetPostprocessor, imagenet_classes_url),
    ], 
    environment=...,
)

client.deploy(client.deployment(making_bundle.get()))
```
#### Forwarders
Another advanced use case is to have a service that makes use of other deployed services. Instead of doing work itself,
such a forwader service does no significant computation on its own. Rather, it serves as a coordination point, spending
most of its time waiting for inference requests to come back.

As an example, consider the `ImagenetService`, but where each of the computation steps is forwarded to an existing 
deployed service:
```python
class RefImagenetService(Service[Dict[str,str], Dict[str,str]], Forwarder):
    
    def __init__(self, preprocess: str, model: str, postprocess: str) -> None:
        super().__init__()
        # deployment_for takes the unique name of an existing deployment
        # and returns a corresponding DeployedService instance 
        self.preprocess: DeployedService = self.client.deployment_for(preprocess)
        # these instances can be used to send requests
        self.model = self.client.deployment_for(model)
        # if deployment_for fails to find a deployed service under the name,
        # then it will raise an Exception
        self.postprocess = self.client.deployment_for(postprocess)
        
    def call(self, request: Dict[str,str]) -> Dict[str,str]:
        image_fut = self.preprocess.send(request)
        preds_fut = self.model.send(image_fut.get())
        response_fut = self.postprocess.send(preds_fut.get())
        return response_fut.get()
```

The new `Forwarder` class's responsibility is to provide a `LaunchClient` so that the user's forwarding service can
obtain `DeployedService` references to the existing deployments. At a minimum, this class would look something like:
```python
class Forwarder:
    def __init__(self) -> None:
        # Special Launch library call that initializes a LaunchClient
        self.client = scale.launch.sumon_client_for_forwarder()        
```

Note that each deployment can have different autoscaling and compute requirements, allowing users to have a 
heterogeneous population of deployed services that is tuned to the specific nature of their workloads. As inference 
courses through the forwarder, the autoscaling system will adapt the number of deployed resources for each. Users
can take advantage of this behavior to benchmark and gain insights about their scaling ability and needs. 

## Batching and Batteries-Included Services

### Batteries Included Model Inference Service
To give users more structure, and benefits, to their services, we encourage the following restricted design pattern:
- a preprocessing step, that turns the request into input for their model
- the normal forward pass inference step of their model: accepting and producing tensors
- a postprocessing step, which transforms the predicted tensors into a well-formatted response

The benefits of such a design for a user are:
- can seamlessly update a service's model and swap out with compatible ones
- request tracing on these stages is automatically included
- can separate CPU (pre+post process) and GPU (model inference) compute and automatically [accelerate with Triton](#triton).

The last point is a **key differentiator for Scale Launch.** The pre- and post-processing Python code can be run in
separate processes from the GPU-bound model inference, allowing one to sidestep the constraints of Python's GIL.

A user must define a [`Processor` class](./launch_api/model_service/types.py), which can turn a request (`I`) into a 
tensor (`B`) and the model's prediction into a response (`O`):
```python
class Processor:
    
    def preprocess(self, request: I) -> B:
        ...

    def postprocess(self, model_output: B) -> O:
        ...
```

A user must also use a type that adheres to the constrint of the [`Model` protocol](./launch_api/model.py):
```python
class Model(Protocol[B]):
    def __call__(self, *args, **kwargs) -> B:
        ...
```
Note that a user **should not extend `Model`**. Rather, they must ensure that their type has a `__call__` method that
accepts either a single input (`B=np.ndarray`) or whose keyword arguments are all `np.ndarray` typed and output returns
a `Dict[str, np.ndarray`]. This means that normal PyTorch and TensorFlow models are compatible with `Model`.
Indeed, **one explicit design objective is to allow a user to drop in their existing ML model into this pattern**. 

The [Batching Interface](#batching-interface) section has more details on the constraints of the tensor type, `B`.
 
### Batching Interface
Along with their [`Model`](./launch_api/model.py), users must define a [`Batcher` class](./launch_api/batching/types.py),
which can turn a list of requests (`List[I]`) into a batch (`B`) and turn a model's  batched predictions into a list of 
responses (`List[O]`):
```python
class Batcher:
    
    def batch(self, requests: List[I]) -> B:
        ...

    def unbatch(self, preds: B) -> List[O]:
        ...
```
Where `B` is either a `np.ndarray` or ` Dict[str, np.ndarray]`. It is the batch array type.
The former case is considered for a simple model: it takes a single array as input and returns a single array as output.
Otherwise, `B` is a set of named arrays. In `batch`, it is the list of keyword array arguments for the model's inference
`__call__` method. And the model is expected to return an output set of named arrays as well.

There are reference implementations for [single arrays here](./launch_api/batching/single_array.py)
and  [named arrays here](./launch_api/batching/named_arrays.py).

Note that the batching interface and accompanying service is _very_ similiar to the 
[pre & post process based service](#batteries-included-model-inference-service). However, with batching, a user
gains the ability to do efficient, high-throughput batch inference. Whether online (via **new accessible batch 
prediction endpoint routes**), or offline (**pre-allocate resources to process `N` items**), by defining methods
to form dense batches for GPU model inference, a user is taking full advantage of their hardware for accelerated 
inference.

Like the "batteries included" service, because of the CPU/GPU compute seperation, a `Batcher`-based service is also 
compatible with [Triton-based model inference](#triton).

### Implementation Details
These batteries-included services have a predefined `call` implementation that cannot be overridden by the user.
There is a [complete implementation included here for processors](./launch_api/model_service/implementation.py) and 
[here for batching](./launch_api/batching/implementation.py). The pseudocode of the `call` method is reproduced below:
```python
def call(self, request: I) -> O:
    try:
        model_input = self.preprocess(request)
    except Exception:
        self.logger.exception(f"Failed to preprocess ({request=})")
        raise
    try:
        pred = self.__call__(model_input)
    except Exception:
        self.logger.exception(
            "Failed inference on request " f"({type(model_input)=}, {request=})"
        )
        raise
    try:
        response = self.postprocess(pred)
    except Exception:
        self.logger.exception(
            "Failed to postprocess prediction "
            f"({type(pred)=}, {type(model_input)=}, {request=})"
        )
        raise
    return response
```

Note that for a `Batcher`, the `call` implementation is essentially identical in structure. The difference being
a call to `self.batch` instead of `self.preprocess` and `self.unbatch ` replacing `self.postprocess`. The types
of `requests`, and return type of `call`, are different too.

## Service Running

### HTTP Runtime for Synchronous Services
Essentially the same as [the HTTP backend in `std-ml-srv`](../std-ml-srv/ml_serve/http_service.py).

### Celery Runtime Asynchronous Services
Essentially the same as [the Celery backend in `std-ml-srv`](../std-ml-srv/ml_serve/celery_service.py).

### Service Loading
Details TDB from [the existing endpoint builder work](../hosted_model_inference/).
The pattern is essentially an interpreter. Users declare a function to load their model service.
The Scale Launch runtime will, upon startup in a deployed container, execute their defined load function.


## Triton
[Triton is an open source ML inference server from NVidia](https://developer.nvidia.com/nvidia-triton-inference-server).
Triton specializes in accelerating model inference both in terms of throughput and latency as compared to running the
same inference code in Python. Unlike Launch, Triton has a narrower scope. Triton provides a tensor-based API for model
inference. That is, clients can request inference by a particular model, but they must provide the n-dimensional
tensor(s) that the model expects _directly_. The client then receives all the output tensors from the model's forward 
pass.

Crucially, Triton requires a **tight coupling** between client and model: the client must know precisely how the model
works in order to carefully prepare its request payloads and properly handle the output predictions. When using Triton,
it is vital to have code surrounding the client that can convert between the data formats required for integration
into a working system as well as handle arbitrary business logic that a production ML inference service requires.

In Launch, using Triton for model inference must be paired with some pre and post processing logic. That logic can
be contained in a single service, implying that the user wants to deploy self-contained compute pods where their
custom Python code is run alongside the accelerated model inference provided by Triton. Alternatively, the user
can make their service a `Forwarder`, delegating the pre and post-processing compute to other services and delegating
the model inference to a hosted Trion model server.

### Triton Runtime
Every Triton-backed model service needs a model repository. This model repository may contain multiple models, each 
of which are formatted in a specific way that Triton expects. The model repository is a directory structure, where each
top-level folder corresponds to a distinct model. Each model directory contains a model configuration file and a 
sub-directory that contains the actual model architecture and weights. Since the directory structure and contents are
important, the natural way to package and manipualate model repositories for Triton is to consider them as `tar` 
archives.

The only other alternative configuration for `trionserver` in Launch is to specify the maximum number of times
Triton should replicate each model across its available GPUs. This `model_named_to_replication` is a simple mapping
of the model name (the name of one of the top-level directories in the model repository) and an integer replication 
factor.

Together, these two pieces of information can fully specify how to start `tritonserver`.
This [simplified `TritonModelServer`](./launch_api/triton.py) illustrates how the runtime starts:
```python
@dataclass
class TritonModelServer(Runtime):

    model_repository_artifact: str
    model_names_to_replication: Optional[Dict[str, int]]

    def start(self) -> None:
        # downloads model artifacts
        dl = download(self.model_repository_artifact)
        # extracts the .tar.gz archive
        local_model_repo = extract(dl)
        # build the command to start triton
        triton_cmd: List[str] = ["tritonserver", "--model-repository", local_model_repo]
        if self.model_names_to_replication is not None:
            for m, i in self.model_names_to_replication.items():
                triton_cmd.extend(
                    ["--rate-limit-resource", f"{m}:{i}"]
                )
            triton_cmd.extend(["--rate-limit", "execution_count"])

        logger.info("Starting Triton...")
        proc = subprocess.Popen(triton_cmd)
        proc.wait()
        raise ValueError("Error: tritonserver quit unexpectedly")
```

### Triton Configuration
As a pure tensor-based model inference framework, Triton requires each model to list the input and output shapes.
Additionally, each input and output must specify the tensors' data type, name, and an optional format. Tensors
are named to support models that accept more than one tensor argument in their forward pass.

To support this format, Launch introduces the `EnrichedTensor` and `TritonTensor`:
```python
class AsPbText:
    def as_pbtext(self) -> str:
        """Formats this datatype's contents as a protobuf text file, which Triton understands."""

class EnrichedTensor:
    data_type: str
    format: Optional[str]
    dims: List[int]


class TritonTensor(EnrichedTensor, AsPbText):
    name: str
    data_type: str
    format: Optional[str]
    dims: List[int]
```
Since Triton reads these configurations from protobuf text config files, we introduce an interface to denote that
certain types support the ability of serializing their contents in this file format for Triton.

Putting these together, a configuration of models for Triton is encapsualted by the following data type:
```python
class TritonModelConfig(AsPbText):
    model_name: str
    platform: str
    backend: Optional[str]
    max_batch_size: int
    dynamic_batching: Optional[float]
    input_shapes: List[TritonTensor]
    output_shapes: List[TritonTensor]
```

Once one has a `TritonModelConfig`, one needs to only also have a corresponding model repository saved as a .tar
archive in order to easily start a triton model service. This information is encapsulated as a `RunnableTritonConfig`:
```python
class RunnableTritonConfig(NamedTuple):
    model_artifact_uri: str
    config: TritonModelConfig
```

### Using the Triton Runtime API
To illustrate how one uses the Launch API for Triton, consider the example of a user with a hotdog image classifier.
This Pytorch model accepts a single input, an image, and produces a single output, the probability that the image
contains a hotdog.

```python
triton_format_model: TritonModelConfig = triton_config_pytorch_model(
    model_name="hot_dog_classifier",
    input_shapes=[EnrichedTensor(data_type="TYPE_INT32", format=None, dims=[255, 255, 3])],
    output_shapes=[EnrichedTensor(data_type="TYPE_FLOAT16", format=None, dims=[1000])],
    max_batch_size=8,
    dynamic_batching=50000,
)
```
The [`triton_config_pytorch_model` function](./launch_api/triton.py) creates a `TritonModelConfiguration` that includes
unique quirks for PyTorch models. PyTorch models's input is passed positionally, not by keyword argument. Therefore,
the names of the input and output tensors for PyTorch models are automatically generated by the TorchScript process.
This function is aware of this convention and populates the `{input,output}_shapes` `name` values appropriately.

Note that a user is responsible for specifying the order, shape (via `dims`), and data type of each input and output 
tensor for their model. Once they have done this, a user can upload their model and configuration as a properly 
formatted Triton model repository:  
```python
model: torch.Module = ...

runnable_configs: Dict[str, RunnableTritonConfig] = make_pytorch_triton_configuration(
    [(model, triton_format_model)],
    s3_client=...,
)
model_name, runnable_config = tuple(runnable_configs.items())
```

The [`make_pytorch_triton_configuration` function](./launch_api/triton.py) creates the necessary directory structure
that Triton expects, places the configuration files and PyTorch model data into the right locations with the right 
names, packages into a compressed tar archive, and uploads the entire bunch to a durable store (e.g. S3, choice tbd.).

NOTE: TensorFlow is also a first-class citizen in Triton and will have full Launch API support. This API example only
covers the PyTorch case, but analogous functions will exist for TensorFlow as well. 

At this point, a user can use this runnable configuration to actually start a local `tritonserver` instance for
development or debugging purposes:
```python
runnable = TritonModelServer(
    model_repository_artifact=runnable_config.model_artifact_uri,
    model_names_to_replication={runnable_config.config.model_name: 1},
)
runnable.start()
```

A user can also deploy a Triton model server from the same configuration information:
```python
client: LaunchClient = ...
ref: DeployedService = client.triton.deploy(runnable_config.model_artifact_uri, 
                                            {runnable_config.config.model_name: 1})

image: PIL.Image.Image = ...
request = triton_format(runnable_config.config, inputs=image)
result_fut: Future = ref.send(request)
preds = triton_format(runnable_config.config, outputs=result_fut.get())
print(preds.shape)
```
Where the `triton_format` function uses the Triton model server configuration information to construct the right
named tensor format that Trtion requires of each model input from simple numpy arrays. Likewise, this function handles 
the reverse process: converting Triton-formatted named output tensors into simplier Python representations.


### Triton Interaction in a Service
Within a single compute pod, a user's `Service` code will use a `TritonClient` to send and receive requests to the 
locally running `tritonserver` instance. The Launch API will provide a special function to resolve this client
when the service starts. For an example, we go back to the simple binary classifier scenario:
```python
class HotDogTriton(Service[dict,dict]):
    
    def __init__(self, triton_config: TritonModelConfig) -> None:
        self.triton_model_config = triton_config
        # Finds the tritonserver running locally
        # Establishes a connection
        self.triton_client = scale.launch.summon_client().summon_local_triton_connection()
        # If no such connection is found, then it raises an Exception
        self.preprocess = ImageDownloaer()
        self.postprocess = HotdogPostprocessor()

    def call(self, request: dict) -> dict:
        image = self.preprocess.call(request)
        preds = self.triton_client.send(
            self.triton_model_config.model_name,
            triton_format(self.triton_model_config, image),
        )
        response = self.postprocess(preds)
        return response
```

Given such a service, one would use the `LaunchClient` to create a service bundle as normal. The main difference comes 
in the `client.deployment` function. Here, a user _must_ specify that they want to also deploy another container in the
pod to run `tritonserver`. They must provide the `RunnableTritonConfig` , which contains a model repository artifact 
URL and a complete specification of the model's input, output, and executable backend. The following code block 
illustrates this API in action:
```python
client: LaunchClient = ...
runnable_triton_config: RunnableTritonConfig = ...

making_bundle = client.create_bundle(
    service=HotDogTriton,
    initialization_args=[runnable_triton_config.config],
    environment=...,
)
bundle = making_bundle.get()

d: Deployment = client.deployment(bundle, 
                                  triton=runnable_triton_config,
                                  runtime=Runtimes.SYNC,
                                  options=...)
client.deploy(d)
```

### Forwarder-Based Triton Integration
Details of this API functionality are still TDB. However, the main difference is that instead of specifying that
`triton=` must deploy in the same pod as the `bundle`, the service that a user deploys will instead use references
to an existing deployed `trionsserver` instance as a `DeploymentService` type. From the user's perspective, their
code still must make calls to the Triton model service (i.e. `self.triton_client.send`). The difference is that
the networked call will be going to a location _outside_ of the local compute node. In the former case, the network
call hits `localhost` and is redirected to the locally-running `tritonserver` container.


## Handling Different Serialized Request Formats
Thus far, this API document has been assuming that the input and output types of a `Service` are always serializable 
and transmissible on any `Runtime`'s networking protocol. Examples thus far have stuck to an "external" data type that
is compatible with JSON data formatting. Thus, implicitly, there has been the notion that the payloads being sent to and
from the `Runtimes` could be considered UTF-8 encoded strings containing JSON data.

However, Launch aims to be more flexible than only supporting services that can send JSON data. For instance, many 
production-grade services send and receive [protocol buffer](https://developers.google.com/protocol-buffers) formatted
data. Launch users should be able to define serializers to convert to and from arbitrary binary formats. To support 
this functionality, we introduce the idea of request and response handlers:
```python
class RequestHandler(Generic[I]):
    def deserialize(self, request: bytes) -> I:
        ...

class ResponseHandler(Generic[O]):
    def serialize(self, response: O) -> bytes:
        ...
```

These handlers can be extended and implemented to support any format. For starters, Launch will provide support for
JSON-encoded data in strings:
```python
class JsonHandler(RequestHandler[Union[dict,list]], ResponseHandler[Union[dict,list]]):
    def deserialize(self, request: bytes) -> Union[dict,list]:
        return json.loads(request.decode('utf-8'))
    
    def serialize(self, response: Union[dict,list]) -> bytes: 
        return json.dumps(response).encode('utf-8')
```
This `JsonHandler` will be the assumed default unless a user specifies otherwise.
Users can override the serializers on creation of the `bundle`:
```python
client.create_bundle(
    ...
    request_handler=...,
    response_handler=...,
)
```

To be specific, the actual service code that every `Runnable` will execute will follow the pattern of:
- deserialize payload
- run service logic
- serialize response

To illustrate further, the actual `Runnable` serving logic will follow 
[the steps of the `RunnableService`'s `call` method](./launch_api/full_service.py):
```python
@dataclass
class RunnableService(Generic[I,O]):
	
    request_handler: RequestHandler[I]
    service: Service[I,O]
    response_handler: ResponseHandler[O]
    
    def call(self, request: bytes) -> bytes:
        # deserialize request payload into format service can handle
        input_ = self.request_handler.deserialize(request)
        # run service logic
        output = self.service.call(input_)            
        # serialize service output for sending over network
        return self.response_handler.serialize(output)
```
