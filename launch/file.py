from typing import List

from pydantic import BaseModel


class UploadFileResponse(BaseModel):
    id: str


class GetFileResponse(BaseModel):
    id: str
    filename: str
    size: int


class ListFilesResponse(BaseModel):
    files: List[GetFileResponse]


class DeleteFileResponse(BaseModel):
    deleted: bool


class GetFileContentResponse(BaseModel):
    id: str
    content: str
