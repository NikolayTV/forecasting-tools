import os
from typing import Final

from openai import AsyncOpenAI

from forecasting_tools.ai_models.model_archetypes.openai_text_model import (
    OpenAiTextToTextModel,
)


class DeepSeekChat(OpenAiTextToTextModel):
    """
    This model sends gpt4o requests to the Metaculus proxy server.
    """

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    _OPENAI_ASYNC_CLIENT = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1/",
        api_key=OPENROUTER_API_KEY,
        max_retries=0,  # Retry is implemented locally
    )

    # See OpenAI Limit on the account dashboard for most up-to-date limit
    MODEL_NAME: Final[str] = "deepseek/deepseek-chat"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 10000
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 40
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 800000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
