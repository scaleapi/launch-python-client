# do not import all endpoints into this module because that uses a lot of memory and stack frames
# if you need the ability to import all endpoints from this module, import them with
# from launch_client.paths.v1_sync_tasks import Api

from launch_client.paths import PathValues

path = PathValues.V1_SYNCTASKS
