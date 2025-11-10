import json
import logging

from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.scan import AIProbabilityRes
from app.services.prompts import AI_CUE_SET, HUMAN_CUE_SET, SYSTEM_INSTRUCTION
from app.utils.scoring import filter_ques, sigmoid

logger = logging.getLogger(__name__)


client = AsyncOpenAI(api_key=settings.openai_api_key)


async def classify_with_openai(fulltext: str | None) -> AIProbabilityRes:
    """OpenAI API를 사용하여 텍스트를 AI 작성 또는 인간 작성으로 분류하고 확률을 계산

    Args:
        fulltext: 분석할 전체 텍스트

    Returns:
        AIProbabilityRes: 분류 레이블과 AI 작성 확률을 포함하는 Pydantic 모델 인스턴스

    Side effects:
        - OpenAI API를 호출하여 네트워크 요청 발생
        - 예외 발생 시 로그 출력 후 기본값 반환
    """
    if not fulltext:
        logger.warning("fulltext가 None이거나 비어있습니다. 기본값을 반환합니다.")
        return AIProbabilityRes(reason="human_like", score=0.5)

    try:
        resp = await client.responses.parse(
            model=settings.openai_model,
            input=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": fulltext},
            ],
            temperature=settings.openai_temperature,
            top_p=0,
        )

        output = json.loads(resp.output_text)

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
        return AIProbabilityRes(score=-1, reason="OpenAI API 응답이 잘못되었습니다.")
    except Exception as e:
        logger.error(f"OpenAI 분류 중 오류 발생: {e}", exc_info=True)
        return AIProbabilityRes(score=-1, reason="OpenAI API 응답이 잘못되었습니다.")
