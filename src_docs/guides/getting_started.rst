Introduction to Scale Launch
============================

Launch has three core concepts:

- ``ModelBundle``
- ``ModelEndpoint``
- ``BatchJob``

``ModelBundle`` represents your model & code, and is what you're deploying;
``ModelEndpoint`` and ``BatchJob`` represent the means of how to deploy it.

ModelBundle
-----------
A ``ModelBundle`` consists of a trained model as well as the surrounding preprocessing and postprocessing code.
Specifically, a ``ModelBundle`` consists of two Python objects, a ``load_predict_fn``, and either a ``model`` or ``load_model_fn``; such that


    load_predict_fn(model)


or


    load_predict_fn(load_model_fn())


returns a function ``predict_fn`` that takes in one argument representing model input,
and outputs one argument representing model output.

Typically, a ``model`` would be a Pytorch nn.Module or Tensorflow model, but can also be any arbitrary Python code.

.. image:: /../src_docs/images/model_bundle.png
    :width: 200px

ModelEndpoint
-------------
A ``ModelEndpoint`` is a deployment of a ``ModelBundle`` that serves inference requests. 
To create a ``ModelEndpoint``, the user must specify various infrastructure-level details,
such as the min & max workers; and the amount of CPU, memory, and GPU resources per worker. A ``ModelEndpoint``
automatically scales the number of workers based on the amount of traffic.

.. image:: /../src_docs/images/model_endpoint.png
    :width: 400px

There are two types of ``ModelEndpoint``:

A ``SyncEndpoint`` takes in requests and immediately returns the response in a blocking manner.
A ``SyncEndpoint`` must always have at least 1 worker.

An ``AsyncEndpoint`` takes in requests and returns an asynchronous response token. The user can later query to monitor
the status of the request. Asynchronous endpoints can scale up from zero,
which make them a cost effective choice for services that are not latency sensitive.

BatchJob
--------
A ``BatchJob`` takes in a ``ModelBundle`` and a list of inputs
to predict on. Once a batch job completes, it cannot be restarted or accept additional requests.
Launch maintains metadata about batch jobs for users to query, even after batch jobs are complete.

Choosing the right inference mode
---------------------------------
Here are some tips for how to choose between ``SyncEndpoint``, ``AsyncEndpoint``, and ``BatchJob`` for deploying your
``ModelBundle``:

A ``SyncEndpoint`` is good if:

- You have strict latency requirements (e.g. on the order of seconds or less).

- You are willing to have resources continually allocated.

An ``AsyncEndpoint`` is good if:

- You want to save on compute costs.

- Your inference code takes a long time to run.

- Your latency requirements are on the order of minutes.

A ``BatchJob`` is good if:

- You know there is a large batch of inputs ahead of time.

- You want to optimize for throughput instead of latency.

Overview of deployment steps
----------------------------
1. Create and upload a ``ModelBundle``. Pass your trained model as well as pre-/post-processing code to
the Scale Launch Python client, and we'll create a model bundle based on the code and store it in our Bundle Store.

2. Create a ``ModelEndpoint``. Pass a ``ModelBundle`` as well as infrastructure settings such as #GPUs to our client.
This provisions resources on Scale's cluster dedicated to your ``ModelEndpoint``.

3. Make requests to the ``ModelEndpoint``. You can make requests through the Python client, or make HTTP requests directly
to Scale.

See the Guides section for more detailed instructions.

.. image:: /../src_docs/images/request_lifecycle.png
    :width: 400px
