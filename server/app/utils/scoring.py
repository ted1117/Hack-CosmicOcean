import math


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
