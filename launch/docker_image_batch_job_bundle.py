import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class CreateDockerImageBatchJobBundleResponse(BaseModel):
    """
    Response Object for creating a Docker Image Batch Job Bundle
    Note: only available for self-hosted mode
    """

    docker_image_batch_job_bundle_id: str
    """ID of the Docker Image Batch Job Bundle"""


class DockerImageBatchJobBundleResponse(BaseModel):
    """
    Response object for a single Docker Image Batch Job Bundle
    Note: only available for self-hosted mode
    """

    id: str
    """ID of the Docker Image Batch Job Bundle"""
    name: str
    """Name of the Docker Image Batch Job Bundle"""
    created_at: datetime.datetime
    """Timestamp of when the Docker Image Batch Job Bundle was created"""
    image_repository: str
    """Short repository name of the underlying docker image"""
    image_tag: str
    """Tag of the underlying docker image"""
    command: List[str]
    """The command to run inside the docker image"""
    env: Dict[str, str]
    """Environment variables to be injected into the docker image"""
    mount_location: Optional[str]
    """Location of a json-formatted file to mount inside the docker image.
        Contents get populated at runtime, and this is the method to change behavior on runtime."""
    cpus: Optional[str]
    """Default number of cpus to give to the docker image"""
    memory: Optional[str]
    """Default amount of memory to give to the docker image"""
    storage: Optional[str]
    """Default amount of disk to give to the docker image"""
    gpus: Optional[int]
    """Default number of gpus to give to the docker image"""
    gpu_type: Optional[str]
    """Default type of gpu, e.g. nvidia-tesla-t4, nvidia-ampere-a10 to give to the docker image"""


class ListDockerImageBatchJobBundleResponse(BaseModel):
    """
    Response object for listing Docker Image Batch Job Bundles.
    Note: only available for self-hosted mode
    """

    docker_image_batch_job_bundles: List[DockerImageBatchJobBundleResponse]
    """A list of 
    [Docker Image Batch Job Bundles](./#launch.docker_image_batch_job_bundle.DockerImageBatchJobBundleResponse)."""
