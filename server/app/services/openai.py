import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.scan import AIProbabilityRes
from app.services.base import BaseAIService
from app.services.prompts import SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


class OpenAIService(BaseAIService):
    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client

    def _request_params(self, fulltext: str) -> dict[str, Any]:
        params = {
            "models": settings.openai_model,
            "input": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": fulltext},
            ],
            "top_p": 0,
        }

        if not settings.openai_model.startswith("gpt-5"):
            params["temperature"] = settings.openai_temperature

        return params

    async def classify(self, fulltext: str) -> AIProbabilityRes:
        try:
            resp = await self.client.responses.parse(**self._request_params(fulltext))

            payload = json.loads(resp.output_text or "")

            if payload:
                return self.build_probability(payload)

            logger.warning("OpenAI API 응답이 비어있습니다. 기본값을 반환합니다.")
            return self.failure_response("OpenAI API 응답이 잘못되었습니다.")

        except json.JSONDecodeError as exc:
            logger.error("JSON 파싱 오류: %s", exc, exc_info=True)
            return self.failure_response("OpenAI API 응답이 잘못되었습니다.")
        except Exception as exc:
            logger.error("OpenAI 분류 중 오류 발생: %s", exc, exc_info=True)
            return self.failure_response("OpenAI API 응답이 잘못되었습니다.")


openai_service = OpenAIService(client=AsyncOpenAI(api_key=settings.openai_api_key))


async def classify_with_openai(fulltext: str | None) -> AIProbabilityRes:
    """OpenAI API를 사용해 텍스트의 AI 작성 확률을 추정한다."""
    if not fulltext:
        logger.warning("fulltext가 None이거나 비어있습니다. 기본값을 반환합니다.")
        return AIProbabilityRes(reason="human_like", score=0.5)

    return await openai_service.classify(fulltext)
