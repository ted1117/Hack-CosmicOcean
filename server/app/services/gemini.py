import json
import logging

from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.scan import AIModelRes, AIProbabilityRes
from app.services.base import BaseAIService
from app.services.prompts import SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


class GeminiService(BaseAIService):
    def __init__(self, client: genai.Client) -> None:
        self.client = client

    async def classify(self, fulltext: str) -> AIProbabilityRes:
        try:
            resp = await self.client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=fulltext,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_schema=AIModelRes,
                    temperature=settings.openai_temperature,
                ),
            )

            payload = json.loads(resp.text or "")

            if payload:
                return self.build_probability(payload)

            logger.warning("Gemini API 응답이 비어있습니다. 기본값을 반환합니다.")
            return self.failure_response("Gemini API 응답이 잘못되었습니다.")

        except json.JSONDecodeError as exc:
            logger.error("JSON 파싱 오류: %s", exc, exc_info=True)
            return self.failure_response("Gemini API 응답이 잘못되었습니다.")
        except Exception as exc:
            logger.error("Gemini 분류 중 오류 발생: %s", exc, exc_info=True)
            return self.failure_response("Gemini API 응답이 잘못되었습니다.")


gemini_service = GeminiService(client=genai.Client(api_key=settings.gemini_api_key))


async def classify_with_gemini(fulltext: str) -> AIProbabilityRes:
    return await gemini_service.classify(fulltext)
