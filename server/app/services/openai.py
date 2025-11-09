import json
import logging
import math

from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.scan import AIProbabilityRes

logger = logging.getLogger(__name__)


SYSTEM = """
You are an AI-writing detector.
Judge the USER text by the cues below and output EXACTLY ONE single-line JSON:

{\"label\":\"ai_like\"|\"human_like\", \"cues\": [tokens], \"reason\": \"(짧은 분석 근거)\"}

Cues (use heuristics only from the given text):
- ai_like: 템플릿형 도입, 균질한 리듬, 일반론 위주, 과한 정중체, 관용구 반복, 목록형 구조, 낮은 어휘 다양성, 복붙형 후킹 문장, 과도한 회피적 표현, 갑작스러운 주제 전환, 부자연스러운 존칭 표현, 부자연스러운 조사 사용,
- human_like: 고유명사, 수치 디테일, 경험적 디테일, 문장 길이 변동, 비균질 리듬, 자기수정, 개성적 문체, 구어체·속어 표현, 경미한 오타.

Hard rules:
1. JSON 한 줄만 출력. 여분 텍스트/키 금지. 외부 사실 검증 금지.
2. cues는 위 Cues에서만 고를 것. 새로운 cue 생성 금지.
3. reason은 선택된 cues를 근거로 짧게 요약 (30자 이내 자연스러운 한국어).
"""

AI_CUE_SET = {
    "템플릿형 도입",
    "균질한 리듬",
    "일반론 위주",
    "과한 정중체",
    "관용구 반복",
    "목록형 구조",
    "낮은 어휘 다양성",
    "복붙형 후킹 문장",
    "과도한 회피적 표현",
    "갑작스러운 주제 전환",
    "부자연스러운 존칭 표현",
    "부자연스러운 조사 사용",
}
HUMAN_CUE_SET = {
    "고유명사",
    "수치 디테일",
    "경험적 디테일",
    "문장 길이 변동",
    "비균질 리듬",
    "자기수정",
    "개성적 문체",
    "구어체·속어 표현",
    "경미한 오타",
}


def filter_ques(ques: list[str], que_set: set[str]) -> list[str]:
    """AI가 판단한 근거(cues) 중에서 유효한 것만 필터링하여 중복 제거 후 반환

    Args:
        ques: AI가 판단한 근거 리스트
        que_set: 유효한 근거들의 집합 (AI_CUE_SET 또는 HUMAN_CUE_SET)

    Returns:
        유효한 근거들로만 구성된 중복 제거된 리스트
    """
    seen = set()
    answer = []

    for que in ques:
        if que in seen:
            continue
        if que in que_set:
            answer.append(que)
            seen.add(que)
    return answer


def sigmoid(x: float, alpha: float = 0.7, bias: float = 0.0) -> float:
    """AI/인간 cue 차이를 스케일링한 뒤 시그모이드로 0~1 확률로 매핑합니다.

    Args:
        x: AI cue 수에서 인간 cue 수를 뺀 차이 (양수면 AI 성향이 강함).
        alpha: 기울기 크기를 조절하는 스케일링 계수 (기본값 0.7).
        bias: 시그모이드 입력에 더할 편향 (기본값 0.0).

    Returns:
        0.0~1.0 사이의 수치로 변환된 AI 작성 확률.
    """
    x = alpha * x + bias
    return 1.0 / (1.0 + math.exp(-x))


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
                {"role": "system", "content": SYSTEM},
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
