# Launch Python Client
```
██╗      █████╗ ██╗   ██╗███╗   ██╗ ██████╗██╗  ██╗
██║     ██╔══██╗██║   ██║████╗  ██║██╔════╝██║  ██║
██║     ███████║██║   ██║██╔██╗ ██║██║     ███████║
██║     ██╔══██║██║   ██║██║╚██╗██║██║     ██╔══██║
███████╗██║  ██║╚██████╔╝██║ ╚████║╚██████╗██║  ██║
╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
```

Moving an ML model from experiment to production requires significant engineering lift.
Scale Launch provides ML engineers a simple Python interface for turning a local code snippet into a production service.
A ML engineer needs to call a few functions from Scale's SDK, which quickly spins up a production-ready service.
The service efficiently utilizes compute resources and automatically scales according to traffic.

Latest API/SDK reference can be found [here](https://scaleapi.github.io/launch-python-client/).

## Deploying your model via Scale Launch

Central to Scale Launch are the notions of a `ModelBundle` and a `ModelEndpoint`.
A `ModelBundle` consists of a trained model as well as the surrounding preprocessing and postprocessing code.
A `ModelEndpoint` is the compute layer that takes in a `ModelBundle`, and is able to carry out inference requests
by using the `ModelBundle` to carry out predictions. The `ModelEndpoint` also knows infrastructure-level details,
such as how many GPUs are needed, what type they are, how much memory, etc. The `ModelEndpoint` automatically handles
infrastructure level details such as autoscaling and task queueing.

Steps to deploy your model via Scale Launch:

1. First, you create and upload a `ModelBundle`.

2. Then, you create a `ModelEndpoint`.

3. Lastly, you make requests to the `ModelEndpoint`.

TODO: link some example colab notebook


## For Developers

Clone from github and install as editable

```
git clone git@github.com:scaleapi/launch-python-client.git
cd launch-python-client
pip3 install poetry
poetry install
```

Please install the pre-commit hooks by running the following command:

```bash
poetry run pre-commit install
```

### Documentation

**Updating documentation:**
We use [Sphinx](https://www.sphinx-doc.org/en/master/) to autogenerate our API Reference from docstrings.

To test your local docstring changes, run the following commands from the repository's root directory:

```
poetry shell
cd src_docs
sphinx-autobuild . ../docs --watch ../launch
```

`sphinx-autobuild` will spin up a server on localhost (port 8000 by default) that will watch for and automatically rebuild a version of the API reference based on your local docstring changes.
