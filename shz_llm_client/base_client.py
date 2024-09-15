from shz_llm_client.schemas import RequestMessage


class BaseLLMClient:
    def __init__(self, api_key, model_id, stream=False, temperature=0.2):
        self._llm_client = None
        self.api_key = api_key
        self._model_id = model_id
        self.stream: bool = stream
        self._temperature: float = temperature
        self._config: dict = {}

    def async_send(self, messages: list[RequestMessage], system_prompt: RequestMessage):
        raise NotImplementedError

    def send(self, messages: list[RequestMessage], system_prompt: RequestMessage):
        raise NotImplementedError

    def _build_payload(
        self, messages: list[RequestMessage], system_prompt: RequestMessage | None
    ) -> dict:
        raise NotImplementedError

    def _make_api_request(self, payload: dict) -> dict:
        raise NotImplementedError

    def _process_response(self, response) -> str:
        raise NotImplementedError

    def _process_stream_response(self, chunk) -> str | dict:
        """
        Output:
            - delta: str
            - usage: dict
        """
        raise NotImplementedError

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        # This validation will force the value to be in that range
        # ref: secure_ai_platform
        # server.mylibs.myopenai.chatbot.valid_temperature
        if not isinstance(value, float):
            raise TypeError("temperature must be a float")

        # Temperature must be between 0 and 1
        self._temperature = max(0, min(1.0, value))

    @property
    def model_id(self) -> str:
        return self._model_id

    @model_id.setter
    def model_id(self, value: str):
        self._model_id = value

    @property
    def config(self) -> dict:
        self._config["model_id"] = self._model_id
        self._config["temperature"] = self._temperature
        self._config["stream"] = self.stream

        return self._config
