from launch.api_client.paths.v1_triggers_trigger_id.delete import ApiFordelete
from launch.api_client.paths.v1_triggers_trigger_id.get import ApiForget
from launch.api_client.paths.v1_triggers_trigger_id.put import ApiForput


class V1TriggersTriggerId(
    ApiForget,
    ApiForput,
    ApiFordelete,
):
    pass
