from abc import ABC, abstractmethod
from typing import Any

from app.schemas.scan import AIProbabilityRes
from app.services.prompts import AI_CUE_SET, HUMAN_CUE_SET
from app.utils.scoring import filter_ques, sigmoid


class BaseAIService(ABC):
    """AI 감지 모델들이 공유하는 공통 유틸과 인터페이스"""

    ai_cues = AI_CUE_SET
    human_cues = HUMAN_CUE_SET

    def build_probability(self, payload: dict[str, Any]) -> AIProbabilityRes:
        """모델이 반환한 payload에서 점수를 계산해 Pydantic 응답으로 변환"""
        cues: list[str] = payload.get("cues", []) or []
        reason = payload.get("reason", "no_reason")

        ai_ques = filter_ques(cues, self.ai_cues)
        human_ques = filter_ques(cues, self.human_cues)

        delta = float(len(ai_ques) - len(human_ques))
        score = round(100 * sigmoid(delta))

        return AIProbabilityRes(score=score, reason=reason)

    def failure_response(self, reason: str) -> AIProbabilityRes:
        """공통 실패 응답 헬퍼"""
        return AIProbabilityRes(score=-1, reason=reason)

    @abstractmethod
    async def classify(self, fulltext: str) -> AIProbabilityRes:
        """입력 텍스트를 분석해 확률 점수를 반환"""
        raise NotImplementedError
