# launch.api_client.model.model_endpoint_resource_state.ModelEndpointResourceState

This is the entity-layer class for the resource settings per worker of a Model Endpoint. Note: in the multinode case, there are multiple \"nodes\" per \"worker\". \"Nodes\" is analogous to a single k8s pod that may take up all the GPUs on a single machine. \"Workers\" is the smallest unit that a request can be made to, and consists of one leader \"node\" and multiple follower \"nodes\" (named \"worker\" in the k8s LeaderWorkerSet definition). cpus/gpus/memory/storage are per-node, thus the total consumption by a \"worker\" is cpus/gpus/etc. multiplied by nodes_per_worker.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | This is the entity-layer class for the resource settings per worker of a Model Endpoint. Note: in the multinode case, there are multiple \&quot;nodes\&quot; per \&quot;worker\&quot;. \&quot;Nodes\&quot; is analogous to a single k8s pod that may take up all the GPUs on a single machine. \&quot;Workers\&quot; is the smallest unit that a request can be made to, and consists of one leader \&quot;node\&quot; and multiple follower \&quot;nodes\&quot; (named \&quot;worker\&quot; in the k8s LeaderWorkerSet definition). cpus/gpus/memory/storage are per-node, thus the total consumption by a \&quot;worker\&quot; is cpus/gpus/etc. multiplied by nodes_per_worker. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**memory** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 
**cpus** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 
**gpus** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 
**nodes_per_worker** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 
**gpu_type** | [**GpuType**](GpuType.md) | [**GpuType**](GpuType.md) |  | [optional] 
**storage** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | [optional] 
**optimize_costs** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

