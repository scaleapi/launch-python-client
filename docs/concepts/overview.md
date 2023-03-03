# Overview

Creating deployments on Launch generally involves three steps:

1. Create and upload a [`ModelBundle`](../model_bundles). Pass your trained model
   as well as pre-/post-processing code to the Scale Launch Python client, and 
   we’ll create a model bundle based on the code and store it in our Bundle Store.

2. Create a [`ModelEndpoint`](../model_endpoints). Pass a ModelBundle as well as
   infrastructure settings such as the desired number of GPUs to our client.
   This provisions resources on Scale’s cluster dedicated to your ModelEndpoint.

3. Make requests to the ModelEndpoint. You can make requests through the Python
   client, or make HTTP requests directly to Scale.
