from collections.abc import Awaitable, Callable

from app.core.config import settings
from app.schemas.scan import AIProbabilityRes
from app.services.gemini import classify_with_gemini
from app.services.openai import classify_with_openai

AIService = Callable[[str], Awaitable[AIProbabilityRes]]


def get_ai_service() -> AIService:
    match settings.ai_provider:
        case "openai":
            return classify_with_openai
        case "gemini":
            return classify_with_gemini
        case _:
            return classify_with_openai
