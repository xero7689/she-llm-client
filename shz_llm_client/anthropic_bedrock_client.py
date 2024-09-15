import json

import aioboto3
import boto3

from .base_client import BaseLLMClient
from .schemas import RequestMessage


class AnthropicBedrockClient(BaseLLMClient):
    """
    Client for Anthropic through AWS Bedrock

    Todo:
    - Handle `aws_region`, `nathropic_version`
    """

    def __init__(
        self,
        model_id,
        stream=False,
        temperature=0.2,
        max_tokens=1000,
        aws_region="us-west-2",
    ):
        super().__init__(
            api_key=None, model_id=model_id, stream=stream, temperature=temperature
        )
        self._model_id = model_id

        self.client = boto3.client(
            service_name="bedrock-runtime", region_name=aws_region
        )

        self._temperature = temperature
        self._max_tokens = max_tokens

    def _build_payload(
        self,
        messages: list[RequestMessage],
        system_prompt: RequestMessage | None = None,
    ) -> dict:
        formatted_messages = []
        for message in messages:
            if len(message.b64_images) == 1:
                formatted_message = {
                    "role": message.role,
                    "content": [],
                }

                for image_item in message.b64_images:
                    formatted_message["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_item.image_type.value}",
                                "data": image_item.b64_string,
                            },
                        }
                    )
                formatted_message["content"].append(
                    {
                        "type": "text",
                        "text": message.content,
                    }
                )

                formatted_messages.append(formatted_message)
            elif len(message.b64_images) > 1:
                formatted_message = {
                    "role": message.role,
                    "content": [],
                }

                for idx, image_item in enumerate(message.b64_images):
                    formatted_message["content"].append(
                        {"type": "text", "text": f"Image {idx+1}:"}
                    )
                    formatted_message["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_item.image_type.value}",
                                "data": image_item.b64_string,
                            },
                        }
                    )
                formatted_message["content"].append(
                    {
                        "type": "text",
                        "text": message.content,
                    }
                )
                formatted_messages.append(formatted_message)
            elif len(message.b64_images) > 20:
                raise ValueError("Claude only supports up to 20 images per request")
            else:
                message_dict = message.dict(exclude={"b64_images"})
                formatted_messages.append(message_dict)

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self._max_tokens,
            "messages": formatted_messages,
        }

        if system_prompt:
            payload["system"] = system_prompt.content

        return payload

    # Async Method
    async def async_send(
        self, messages: list[RequestMessage], system_prompt: RequestMessage
    ):
        payload = self._build_payload(messages, system_prompt)

        aio_session = aioboto3.Session()
        async with aio_session.client(
            "bedrock-runtime", region_name="us-west-2"
        ) as aio_client:
            if self.stream:
                response = await aio_client.invoke_model_with_response_stream(
                    body=json.dumps(payload), modelId=self._model_id
                )
                async for event in response.get("body"):
                    chunk = json.loads(event["chunk"]["bytes"])
                    yield self._process_stream_response(chunk)
            else:
                response = await aio_client.invoke_model(
                    body=payload, modelId=self._model_id
                )
                yield self._process_response(response)

    # Sync Method
    def _stream_response_generator(self, response):
        for event in response.get("body"):
            yield self._process_response(event)

    def _make_api_request(self, payload: dict) -> dict:
        payload = json.dumps(payload)
        if self.stream:
            return self.client.invoke_model_with_response_stream(
                body=payload, modelId=self._model_id
            )
        return self.client.invoke_model(body=payload, modelId=self._model_id)

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
    def _process_response(self, response: dict) -> str:
        if self.stream:
            chunk = json.loads(response["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                return chunk["delta"]["text"]
            else:
                return ""
        else:
            body = response.get("body", None)
            if body is None:
                return ""
            else:
                response_body = json.loads(body.read())
                contents = response_body.get("content", [])
                text = contents[0].get("text", "")
                return text

    def _process_stream_response(self, chunk) -> dict:
        if chunk["type"] == "content_block_delta":
            return {
                "delta": chunk["delta"]["text"],
                "type": "delta",
            }
        elif chunk["type"] == "message_stop":
            usage = chunk["amazon-bedrock-invocationMetrics"]
            return {
                "delta": "",
                "input_tokens": usage["inputTokenCount"],
                "output_tokens": usage["outputTokenCount"],
                "total_tokens": usage["inputTokenCount"] + usage["outputTokenCount"],
                "type": "stop",
            }
        else:
            return {"delta": "", "type": "delta"}
