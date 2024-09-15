# SHZ LLM Client

This project serves as a universal Large Language Model (LLM) interface for the SHZ-GPT initiative.

It integrates with popular LLM client SDKs, including `OpenAI`, `Google Gemini`, `Anthropic Claude`, and others. By providing a consistent interface, it allows developers to quickly and easily switch between different language models when building applications, streamlining the development process and enhancing flexibility.

## Supported Client
- OpenAI
- Anthropic Claude (through AWS Bedrock)
- Google Gemini
- Perplexity

## Usage
```python
from shz_llm_client import OpenAIClient, RequestMessage

client = OpenAIClient(api_key=api_key, stream=False)


system_prompt = RequestMessage(
    role="system", content="You are a helpful assistant."
)
messages = [RequestMessage(role="user", content="Hello, GPT!")]

response = client.send(messages, system_prompt)
```
