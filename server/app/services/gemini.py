import json
import logging

from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.scan import AIModelRes, AIProbabilityRes
from app.services.prompts import AI_CUE_SET, HUMAN_CUE_SET, SYSTEM_INSTRUCTION
from app.utils.scoring import filter_ques, sigmoid

logger = logging.getLogger(__name__)


client = genai.Client(api_key=settings.gemini_api_key)


async def classify_with_gemini(fulltext: str) -> AIProbabilityRes:
    try:
        resp = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=fulltext,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_schema=AIModelRes,
                temperature=settings.openai_temperature,
            ),
        )

        output = json.loads(resp.text or "")

        if output:
            label = output.get("label", "uncertain")
            reason = output.get("reason", "no_reason")
            que_list = output.get("cues", [])

            ai_ques = filter_ques(que_list, AI_CUE_SET)
            human_ques = filter_ques(que_list, HUMAN_CUE_SET)

            n_ai = len(ai_ques)
            n_human = len(human_ques)

            x = float(n_ai - n_human)
            prob = round(100 * sigmoid(x))

            return AIProbabilityRes(score=prob, reason=reason)
        else:
            logger.warning("OpenAI API 응답이 비어있습니다. 기본값을 반환합니다.")
            return AIProbabilityRes(
                score=-1, reason="OpenAI API 응답이 잘못되었습니다."
            )

    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}", exc_info=True)
        return AIProbabilityRes(score=-1, reason="Gemini API 응답이 잘못되었습니다.")
    except Exception as e:
        logger.error(f"OpenAI 분류 중 오류 발생: {e}", exc_info=True)
        return AIProbabilityRes(score=-1, reason="Gemini API 응답이 잘못되었습니다.")
