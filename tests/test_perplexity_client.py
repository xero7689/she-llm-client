import os

import pytest
from faker import Faker
from shz_llm_client import PerplexityClient, RequestMessage

fake = Faker()

STATE_PROMPT = "Hello, please state 'I am an LLM client' without adding any additional commentary or words from your end."

STATE_PROMPT_EXPECTED_RESPONSE = "I am an LLM client"

api_key = os.environ["PERPLEXITY_API_KEY"]


@pytest.mark.skipif(
    "TEST_EXTERNAL_API" not in os.environ,
    reason="TEST_EXTERNAL_API is not set",
)
@pytest.mark.django_db
def test_sync_perplexity_client_without_stream():
    client = PerplexityClient(api_key=api_key, stream=False)

    system_prompt = RequestMessage(
        role="system", content="You are a helpful assistant."
    )
    messages = [RequestMessage(role="user", content=STATE_PROMPT)]

    response = client.send(messages, system_prompt)

    assert isinstance(response, str)
    assert STATE_PROMPT_EXPECTED_RESPONSE in response
