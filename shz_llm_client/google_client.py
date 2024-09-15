import base64
import logging
from io import BytesIO

import PIL.Image
from google import generativeai as genai

from .base_client import BaseLLMClient
from .schemas import RequestMessage

logger = logging.getLogger(__name__)


class GoogleClient(BaseLLMClient):
    """
    Google Client

    Gemini offers two text completion API:
        - `generate_content`: Like other text completion API, it generates text based on the given prompts list.
        - `start_chat`

    While `start_chat` maintains the chat history itself, it is best to reuse it
    through out the whole chat session, instead of creating a new session for each message request.
    This client temporarily uses `generate_content` API, to perform the request like the traditional text completion API.
    """

    def __init__(
        self, api_key, model_id="gemini-1.5-flash", stream=False, temperature=0.2
    ):
        super().__init__(api_key, model_id, stream, temperature)
        genai.configure(api_key=api_key)
        self._model_id = model_id
        self._client = genai.GenerativeModel(model_name=model_id)

    def _get_client_with_sys_prompt(self, system_instruction: str):
        return genai.GenerativeModel(
            model_name=self._model_id, system_instruction=system_instruction
        )

    def _build_payload(
        self, messages: list[RequestMessage], system_prompt: RequestMessage | None
    ):
        text_only_messages = []
        image_contain_messages = []
        for message in messages:
            if message.b64_images:
                image_contain_messages.append(message)
            else:
                text_only_messages.append(message)

        formatted_messages = []
        if not image_contain_messages:
            for message in messages:
                role = message.role
                if role == "assistant":
                    role = "model"
                content = message.content
                tmp_prompt = {"role": role, "parts": content}
                formatted_messages.append(tmp_prompt)
        else:
            for message in messages:
                role = message.role
                if role == "assistant":
                    role = "model"
                if message.b64_images:
                    formatted_messages.append(f"{role}: {message.content}")
                    for image_item in message.b64_images:
                        b64_image_str = image_item.b64_string
                        img_bytes = base64.b64decode(b64_image_str)
                        image = PIL.Image.open(BytesIO(img_bytes))
                        formatted_messages.append(image)
                else:
                    formatted_messages.append(f"{role}: {message.content}")

        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": 800,
        }

        payload = {
            "contents": formatted_messages,
            "generation_config": generation_config,
        }

        if self.stream:
            payload["stream"] = True

        return payload

    # Async Method
    async def async_send(
        self, messages: list[RequestMessage], system_prompt: RequestMessage
    ):
        payload = self._build_payload(messages, system_prompt)

        # Gemini needs to specific system_prompt at client level
        if system_prompt.content:
            client = self._get_client_with_sys_prompt(system_prompt.content)
        else:
            client = self._client

        response = await client.generate_content_async(**payload)

        if self.stream:
            async for chunk in response:
                yield self._process_stream_response(chunk)

            # genai-0.7.2 doesn't have information of whether the response is stopped or not
            # We manually add the stop message here for token count
            yield {
                "delta": "",
                "input_tokens": chunk.usage_metadata.prompt_token_count,
                "output_tokens": chunk.usage_metadata.candidates_token_count,
                "total_tokens": chunk.usage_metadata.total_token_count,
                "type": "stop",
            }
        else:
            yield self._process_response(response)

    # Sync Method
    def _stream_response_generator(self, response):
        for chunk in response:
            yield self._process_response(chunk)

    def send(self, messages: list[dict], system_prompt: dict):
        payload = self._build_payload(messages, system_prompt)

        if system_prompt.content:
            client = self._get_client_with_sys_prompt(system_prompt.content)
        else:
            client = self._client

        response = client.generate_content(**payload)

        if self.stream:
            return self._stream_response_generator(response)
        else:
            return self._process_response(response)

    #
    # Process Response
    #
    def _process_response(self, response) -> str:
        candidate = response.candidates[0]
        try:
            text = candidate.content.parts[0].text
        except (AttributeError, IndexError):
            if candidate.finish_reason == 2:
                text = "<Candidate reached MAX_TOKENS>"
            elif candidate.finish_reason == 3:
                text = "<Candidate flagged for safety reasons>"
            elif candidate.finish_reason == 4:
                text = "<Candidate flagged for recitation reasons>"
            else:
                text = "<Candidate for other reason>"
            logger.warning(
                f"[{self._model_id}] Response Candidate stop warning: {text}"
            )
            logger.warning(candidate)
            text = ""

        return text

    def _process_stream_response(self, chunk) -> dict:
        candidate = chunk.candidates[0]
        try:
            text = candidate.content.parts[0].text
        except (AttributeError, IndexError):
            if candidate.finish_reason == 2:
                text = "<Candidate reached MAX_TOKENS>"
            elif candidate.finish_reason == 3:
                text = "<Candidate flagged for safety reasons>"
            elif candidate.finish_reason == 4:
                text = "<Candidate flagged for recitation reasons>"
            else:
                text = "<Candidate for other reason>"
            logger.warning(
                f"[{self._model_id}] Response Candidate stop warning: {text}"
            )
            logger.warning(candidate)
            text = ""
        return {
            "delta": text,
            "type": "delta",
        }


class GoogleLegacyClient(BaseLLMClient):
    """
    Google Legacy Client
    """

    def __init__(self, api_key, model_id="models/text-bison-001", temperature=0.2):
        super().__init__(api_key, model_id)
        self._model_id = model_id
        self._client = genai.configure(api_key=api_key)
        self._temperature = temperature

    def send(self, messages: list[dict], system_prompt: dict):
        payload = self._build_payload(messages, system_prompt)
        response = self._make_api_request(payload)
        return self._process_response(response)

    def _build_payload(self, messages: list[dict], system_prompt: dict):
        prompt_text = f"{system_prompt['role']}: {system_prompt['content']}\n"

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            prompt_text += f"{role}: {content}\n"

        payload = {
            "model": self._model_id,
            "prompt": prompt_text,
            "temperature": 0,
            "max_output_tokens": 800,  # Maximun length of the response
        }

        return payload

    def _make_api_request(self, payload: dict):
        return genai.generate_text(**payload)

    def _process_response(self, response) -> str:
        # Todo: Add a check to see if the response is valid
        cleaned_text = response.result.replace("\nassistant: ", "")
        return cleaned_text
