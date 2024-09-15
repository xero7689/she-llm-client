from .anthropic_client import AnthropicClient
from .google_client import GoogleClient
from .openai_client import OpenAIClient
from .perplexity_client import PerplexityClient


class LLMClientFactory:
    @staticmethod
    def create_client(vendor_name: str, model_id: str, api_key: str, **kwargs):
        if vendor_name.lower() == "openai":
            return OpenAIClient(api_key=api_key, model_id=model_id, **kwargs)
        elif vendor_name.lower() == "google":
            return GoogleClient(api_key=api_key, model_id=model_id, **kwargs)
        elif vendor_name.lower() == "anthropic":
            return AnthropicClient(model_id=model_id, **kwargs)
        elif vendor_name.lower() == "perplexity":
            return PerplexityClient(api_key=api_key, model_id=model_id, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM Vendor: {vendor_name}")
