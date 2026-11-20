"""Helpers for turning user-provided images into model content blocks.

The agent loop itself is text based: a task is rendered into a user message and
the model responds with bash tool calls.  To support image input we represent
images as OpenAI-compatible ``image_url`` content blocks in that first user
message.  Images may be supplied as local paths, ``http(s)`` URLs, data URLs,
raw bytes, or small dictionaries (``{"path": ...}``/``{"url": ...}``).
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Keep individual attachments reasonably small.  GUI requests allow multiple
# images, so this also keeps the overall JSON payload bounded.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

_IMAGE_MIME_BY_SUFFIX = {
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


class ImageInputError(ValueError):
    """Raised when an image input cannot be converted into a usable URL."""


def _data_url_for_bytes(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _mime_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_MIME_BY_SUFFIX:
        return _IMAGE_MIME_BY_SUFFIX[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


def image_to_url(image: Any, *, max_bytes: int = MAX_IMAGE_BYTES) -> str:
    """Normalize one image input to a URL usable by multimodal chat APIs.

    Local files are inlined as data URLs so the model can access them without
    requiring an additional upload step.  Remote URLs and data URLs are passed
    through unchanged.
    """

    if isinstance(image, Path):
        path = image.expanduser()
        if not path.is_file():
            raise ImageInputError(f"Image file does not exist: {path}")
        data = path.read_bytes()
        if len(data) > max_bytes:
            raise ImageInputError(f"Image is too large ({len(data)} bytes); limit is {max_bytes} bytes")
        return _data_url_for_bytes(data, _mime_for_path(path))

    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if len(data) > max_bytes:
            raise ImageInputError(f"Image is too large ({len(data)} bytes); limit is {max_bytes} bytes")
        return _data_url_for_bytes(data, "image/png")

    if isinstance(image, dict):
        for key in ("url", "data_url", "path"):
            if image.get(key):
                return image_to_url(image[key], max_bytes=max_bytes)
        raise ImageInputError("Image dictionaries must contain a 'url', 'data_url', or 'path' value")

    if not isinstance(image, str):
        raise ImageInputError(f"Unsupported image input type: {type(image).__name__}")

    value = image.strip()
    if not value:
        raise ImageInputError("Image input cannot be empty")
    if value.startswith("data:image/"):
        return value
    if value.startswith(("http://", "https://")):
        return value

    path = Path(value).expanduser()
    if path.is_file():
        return image_to_url(path, max_bytes=max_bytes)

    raise ImageInputError(
        f"Could not load image {value!r}: expected an existing file, an http(s) URL, or a data:image URL"
    )


def normalize_images(images: Iterable[Any] | None) -> list[str]:
    """Normalize a sequence of image inputs into model-ready URLs."""

    if images is None:
        return []
    if isinstance(images, (str, Path, bytes, bytearray, memoryview, dict)):
        images = [images]
    return [image_to_url(image) for image in images]


def build_multimodal_content(text: str, image_urls: Iterable[str] | None) -> str | list[dict]:
    """Return a plain string when there are no images, or multimodal blocks."""

    urls = [str(url) for url in (image_urls or []) if url]
    if not urls:
        return text
    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.extend({"type": "image_url", "image_url": {"url": url}} for url in urls)
    return blocks


def to_responses_api_content(content: Any) -> Any:
    """Convert OpenAI chat content blocks to Responses API input blocks."""

    if not isinstance(content, list):
        return content
    converted: list[Any] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            converted.append({"type": "input_text", "text": item.get("text", "")})
        elif isinstance(item, dict) and item.get("type") == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            converted.append({"type": "input_image", "image_url": image_url})
        else:
            converted.append(item)
    return converted
