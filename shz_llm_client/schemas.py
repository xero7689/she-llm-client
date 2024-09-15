from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator


def _image_type_validator(value: Any) -> str:
    if value == "jpg":
        return "jpeg"
    return value


class SupportedImageTypes(str, Enum):
    jpeg = "jpeg"
    png = "png"
    gif = "gif"
    webp = "webp"


class Base64ImageItem(BaseModel):
    b64_string: str
    image_type: Annotated[SupportedImageTypes, BeforeValidator(_image_type_validator)]
    image_name: str = ""
    image_id: str = ""


class RequestMessage(BaseModel):
    content: str
    role: str
    b64_images: list[Base64ImageItem] = []
