import os
import asyncio
import time
from dotenv import load_dotenv

import pytest
from faker import Faker
from shz_llm_client import Base64ImageItem, OpenAIClient, RequestMessage
from shz_llm_client.vision import image_to_base64

load_dotenv()
fake = Faker()

STATE_PROMPT = "Hello, please state 'I am an LLM client' without adding any additional commentary or words from your end."

STATE_PROMPT_EXPECTED_RESPONSE = "I am an LLM client"

api_key = os.environ["OPENAI_API_KEY"]
test_model_id = "gpt-4o-mini"


@pytest.mark.skipif(
    "TEST_EXTERNAL_API" not in os.environ,
    reason="TEST_EXTERNAL_API is not set",
)
@pytest.mark.django_db
def test_sync_openai_client_without_stream():
    client = OpenAIClient(api_key=api_key, stream=False)

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
    client = OpenAIClient(api_key=api_key, model_id=test_model_id, stream=False)

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
    assert "starry night" in response.lower()


@pytest.mark.skipif(
    "TEST_EXTERNAL_API" not in os.environ,
    reason="TEST_EXTERNAL_API is not set in the environment variables",
)
def test_multiple_images_vision_support():
    client = OpenAIClient(api_key=api_key, model_id=test_model_id, stream=False)

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
    client = OpenAIClient(api_key=api_key, model_id=test_model_id, stream=False)

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
    client = OpenAIClient(api_key=api_key, stream=True)

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


@pytest.mark.skipif(
    ("TEST_EXTERNAL_API" not in os.environ)
    and ("PRESSURE_TEST_EXTERNAL_API" not in os.environ),
    reason="TEST_EXTERNAL_API is not set",
)
@pytest.mark.asyncio
async def test_multi_client_async_send():
    """The pressure test for multiple clients sending requests to the OpenAI API."""

    async def send_request(client, client_id, messages, system_prompt):
        """
        Send a request to the client and print the response.
        """
        start_time = time.time()  # Start the timer
        response = []
        async for resp in client.async_send(messages, system_prompt):
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Client {client_id}: {resp}\t<{elapsed_time:.2f}s>")
            response.append(resp["delta"])
            start_time = time.time()
        return (client, "".join(response))

    ethical_ai_questions = [
        "In what ways do you think potential biases in AI algorithms can impact patient treatment outcomes, and how can we mitigate these biases to ensure equitable healthcare delivery across diverse patient populations?",
        "Given the importance of patient privacy and data security, what specific strategies should healthcare organizations implement to protect sensitive information when using AI technologies for diagnosis and treatment?",
        "Which ethical frameworks are most relevant for guiding the development and deployment of artificial intelligence in medical applications, and how can these frameworks ensure that the rights and dignity of patients are upheld?",
        "When AI systems make clinical decisions that affect patient care, what principles of accountability should be established to determine who is responsible for the outcomes, especially in cases of misdiagnosis or error?",
        "Should patients have the right to be informed when they are receiving care from AI systems, and what approaches should healthcare providers take to communicate this information transparently to ensure patient understanding and trust?",
        "What proactive measures can be adopted to prevent discrimination against marginalized and underrepresented groups in the context of AI healthcare solutions, particularly in terms of access to services and treatment recommendations?",
        "How can we effectively design AI systems in healthcare to support and enhance the work of human healthcare providers rather than replace them, fostering a collaborative approach to patient care and treatment decisions?",
        "In the context of using AI in patient care, what role does informed consent play, and how should healthcare professionals ensure that patients fully understand the implications of AI technologies on their health and treatment options?",
        "How can healthcare organizations evaluate the trustworthiness and reliability of AI tools and solutions that are being implemented in clinical settings, and what criteria should be considered to ensure the safety of patients?",
        "What steps can be taken to ensure that diverse stakeholders, including patients, healthcare providers, and ethicists, are involved in discussions about ethical AI in healthcare to promote inclusive decision-making and address potential ethical dilemmas?",
    ]
    system_prompt = RequestMessage(role="system", content="")

    client_number = 2

    clients = [OpenAIClient(api_key=api_key, stream=True) for _ in range(client_number)]
    client_message_hisotry = {client: [] for client in clients}

    for idx, question in enumerate(ethical_ai_questions):
        print(f"===Test Question {idx}===")

        jobs = []
        user_message = RequestMessage(role="user", content=question)
        for idx, client in enumerate(clients):
            client_message_hisotry[client].append(user_message)
            jobs.append(
                send_request(client, idx, client_message_hisotry[client], system_prompt)
            )

        results = await asyncio.gather(*jobs)

        for client, response in results:
            client_message_hisotry[client].append(
                RequestMessage(role="assistant", content=response)
            )
