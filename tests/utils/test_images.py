import base64

import pytest

from minisweagent.utils.images import (
    ImageInputError,
    build_multimodal_content,
    image_to_url,
    normalize_images,
    to_responses_api_content,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_local_image_is_inlined_as_data_url(tmp_path):
    image = tmp_path / "pixel.png"
    image.write_bytes(PNG_BYTES)

    url = image_to_url(image)

    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG_BYTES


def test_remote_and_data_urls_pass_through():
    assert image_to_url("https://example.com/screenshot.png") == "https://example.com/screenshot.png"
    data_url = "data:image/png;base64,iVBORw0KGgo="
    assert image_to_url(data_url) == data_url


def test_structured_and_raw_image_inputs(tmp_path):
    image = tmp_path / "pixel.png"
    image.write_bytes(PNG_BYTES)

    assert image_to_url({"path": str(image)}).startswith("data:image/png;base64,")
    assert image_to_url({"url": "https://example.com/cat.jpg"}) == "https://example.com/cat.jpg"
    assert image_to_url(PNG_BYTES).startswith("data:image/png;base64,")
    assert normalize_images("https://example.com/a.png") == ["https://example.com/a.png"]


def test_invalid_image_input_raises(tmp_path):
    with pytest.raises(ImageInputError):
        image_to_url("")
    with pytest.raises(ImageInputError):
        image_to_url(tmp_path / "missing.png")
    with pytest.raises(ImageInputError):
        image_to_url({"unexpected": "value"})


def test_build_multimodal_content():
    assert build_multimodal_content("hello", []) == "hello"
    assert build_multimodal_content("hello", ["https://example.com/a.png"]) == [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]


def test_to_responses_api_content():
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]
    assert to_responses_api_content(content) == [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": "https://example.com/a.png"},
    ]
