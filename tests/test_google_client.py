import os

import pytest
from faker import Faker
from shz_llm_client import Base64ImageItem, GoogleClient, RequestMessage
from shz_llm_client.vision import image_to_base64

from dotenv import load_dotenv

load_dotenv()
fake = Faker()


STATE_PROMPT = "Hello, please state 'I am an LLM client' without adding any additional commentary or words from your end."

STATE_PROMPT_EXPECTED_RESPONSE = "I am an LLM client"

api_key = os.environ["GOOGLE_API_KEY"]
model_id = "gemini-1.5-pro"


@pytest.mark.skipif(
    "TEST_EXTERNAL_API" not in os.environ,
    reason="TEST_EXTERNAL_API is not set",
)
def test_google_client():
    client = GoogleClient(api_key=api_key)

    system_prompt = RequestMessage(
        role="system", content="You are a helpful assistant."
    )
    messages = [RequestMessage(role="user", content=STATE_PROMPT)]

    response = client.send(messages, system_prompt)
    assert isinstance(response, str)

    assert STATE_PROMPT_EXPECTED_RESPONSE in response


@pytest.mark.skipif(
    "TEST_EXTERNAL_API" not in os.environ,
    reason="TEST_EXTERNAL_API is not set in the environment variables",
)
def test_vision_support():
    client = GoogleClient(api_key=api_key, model_id=model_id, stream=False)

    system_prompt = RequestMessage(
        role="system", content="You are a helpful assistant."
    )

    image_path = "./tests/images/starry-night.jpg"
    b64_image = image_to_base64(image_path)
    image_item = Base64ImageItem(b64_string=b64_image, image_type="jpg")

    messages = [
        RequestMessage(
            role="user", content="What is this image about?", b64_images=[image_item]
        ),
    ]

    response = client.send(messages, system_prompt)
    assert isinstance(response, str)
    print(response)
    assert "starry night" in response.lower()


@pytest.mark.skipif(
    "TEST_EXTERNAL_API" not in os.environ,
    reason="TEST_EXTERNAL_API is not set in the environment variables",
)
def test_multiple_images_vision_support():
    client = GoogleClient(api_key=api_key, model_id=model_id, stream=False)

    system_prompt = RequestMessage(
        role="system", content="You are a helpful assistant."
    )

    image_path_1 = "./tests/images/starry-night.jpg"
    b64_image_1 = image_to_base64(image_path_1)
    image_item_1 = Base64ImageItem(b64_string=b64_image_1, image_type="jpg")

    image_path_2 = "./tests/images/vanGoh.jpg"
    b64_image_2 = image_to_base64(image_path_2)
    image_item_2 = Base64ImageItem(b64_string=b64_image_2, image_type="jpg")

    messages = [
        RequestMessage(
            role="user",
            content="Do they have relation?",
            b64_images=[image_item_1, image_item_2],
        ),
    ]

    response = client.send(messages, system_prompt)
    assert isinstance(response, str)


@pytest.mark.skipif(
    "TEST_EXTERNAL_API" not in os.environ,
    reason="TEST_EXTERNAL_API is not set in the environment variables",
)
def test_complex_messages_structure_with_vision_support():
    client = GoogleClient(api_key=api_key, model_id=model_id, stream=False)

    system_prompt = RequestMessage(
        role="system", content="You are a helpful assistant."
    )

    image_path_1 = "./tests/images/starry-night.jpg"
    b64_image_1 = image_to_base64(image_path_1)
    image_item_1 = Base64ImageItem(b64_string=b64_image_1, image_type="jpg")

    image_path_2 = "./tests/images/vanGoh.jpg"
    b64_image_2 = image_to_base64(image_path_2)
    image_item_2 = Base64ImageItem(b64_string=b64_image_2, image_type="jpg")

    messages = [
        RequestMessage(
            role="user",
            content="What time in this paint?",
            b64_images=[image_item_1],
        ),
        RequestMessage(
            role="assistant",
            content="It is at night.",
        ),
        RequestMessage(
            role="user",
            content="What about this one? Do they have relation?",
            b64_images=[image_item_2],
        ),
    ]

    response = client.send(messages, system_prompt)
    assert isinstance(response, str)
    print(response)


@pytest.mark.skipif(
    "TEST_EXTERNAL_API" not in os.environ,
    reason="TEST_EXTERNAL_API is not set",
)
@pytest.mark.asyncio
async def test_async_stream_response():
    api_key = os.environ["GOOGLE_API_KEY"]
    client = GoogleClient(api_key=api_key, stream=True)

    system_prompt = RequestMessage(
        role="system", content="You are a helpful assistant."
    )
    messages = [RequestMessage(role="user", content=STATE_PROMPT)]

    async for resp in client.async_send(messages, system_prompt):
        print(resp)

    # Verify the last response contains the expected usage information
    last_resp = resp
    assert last_resp["type"] == "stop"
    for k in ["input_tokens", "output_tokens", "total_tokens", "type"]:
        assert k in last_resp
