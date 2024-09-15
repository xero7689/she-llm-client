import logging

import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .base_client import BaseLLMClient
from .schemas import RequestMessage

logger = logging.getLogger(__name__)

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"


class PerplexityClient(BaseLLMClient):
    """
    Perplexity's API is OpenAI Client compatible, we directly inherit OpenAIClient
    """

    def __init__(
        self,
        api_key,
        model_id="llama-3-sonar-large-32k-online",
        stream=False,
        temperature=0.2,
    ):
        super().__init__(api_key, model_id, stream, temperature)

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=PERPLEXITY_BASE_URL,
        )

        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=PERPLEXITY_BASE_URL,
        )

    def _build_payload(
        self,
        messages: list[RequestMessage],
        system_prompt: RequestMessage | None = None,
    ):
        """
        Build payload for Perplexity's API

        Perplexity's Online Model doesn't support system prompt currently
        And it also needs the roles to be alternating in the message list

        We directly set the system prompt at the top of the latest user message
        """

        formatted_messages = []

        for message in messages:
            if message.b64_images:
                formatted_message = {
                    "role": message.role,
                    "content": [],
                }

                formatted_message["content"].append(
                    {
                        "type": "text",
                        "text": message.content,
                    }
                )

                b64_images = message.b64_images
                image_payload = {}
                for image_item in b64_images:
                    image_payload["type"] = "image_url"
                    image_payload["image_url"] = {
                        "url": f"data:image/{image_item.image_type.value};base64,{image_item.b64_string}"
                    }
                    formatted_message["content"].append(image_payload)
                formatted_messages.append(formatted_message)
            else:
                formatted_messages.append(message.dict(exclude={"images"}))

        if system_prompt and system_prompt.content:
            latest_usre_mesesage = formatted_messages[-1]
            original_user_content = latest_usre_mesesage["content"]
            latest_usre_mesesage["content"] = (
                f"{system_prompt.content}\n{original_user_content}"
            )

        payload = {
            "model": self._model_id,
            "messages": formatted_messages,
            "temperature": self.temperature,
        }

        if self.stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        return payload

    #
    # Async Method
    #
    async def _async_make_api_request(self, payload: dict):
        response = await self.async_client.chat.completions.create(**payload)
        return response

    async def async_send(
        self, messages: list[RequestMessage], system_prompt: RequestMessage
    ):
        payload = self._build_payload(messages, system_prompt)
        response = await self._async_make_api_request(payload)

        if self.stream:
            async for chunk in response:
                yield self._process_stream_response(chunk)
        else:
            # Todo:
            # Test none-stream mode code in async mode
            yield self._process_response(response)

    #
    # Sync Method
    #
    def _stream_response_generator(self, response):
        for chunk in response:
            yield self._process_stream_response(chunk)

    def _make_api_request(self, payload: dict):
        response = self.client.chat.completions.create(**payload)
        return response

    def send(self, messages: list[RequestMessage], system_prompt: RequestMessage):
        payload = self._build_payload(messages, system_prompt)
        response = self._make_api_request(payload)

        if self.stream:
            return self._stream_response_generator(response)
        else:
            return self._process_response(response)

    #
    # Process Response
    #
    def _process_response(self, response: ChatCompletion) -> str:
        if len(response.choices) == 0:
            return ""

        try:
            content = response.choices[0].message.content
        except AttributeError:
            logger.warning(f"Content not found content in response: {response}")
            return ""

        return content

    def _process_stream_response(self, chunk: ChatCompletionChunk) -> str | dict:
        choice = chunk.choices[0]
        if choice.finish_reason not in ["stop", None]:
            logger.warning(f"{chunk.id}: Finish Reason: {choice.finish_reason}")

        if chunk.choices[0].finish_reason != "stop":
            return {
                "delta": chunk.choices[0].delta.content,
                "type": "delta",
            }
        else:
            return {
                "delta": chunk.choices[0].delta.content,
                "input_tokens": chunk.usage.prompt_tokens,
                "output_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
                "type": "stop",
            }
