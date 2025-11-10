import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.deps_ai import AIService, get_ai_service
from app.schemas.scan import AIProbabilityRes, ScanErrorRes, URLInputReq
from app.services.extractor import extract_article

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/score",
    response_model=AIProbabilityRes,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ScanErrorRes,
            "description": "본문을 추출할 수 없습니다.",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "model": ScanErrorRes,
            "description": "본문 추출 또는 AI 분류 중 오류가 발생했습니다.",
        },
    },
)
async def get_ai_probability_from_url(
    req: URLInputReq, ai_service: AIService = Depends(get_ai_service)
) -> AIProbabilityRes:
    url = str(req.url)

    try:
        fulltext: str | None = await extract_article(url)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"본문 추출 중 오류 발생: {url}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="본문 추출 중 오류가 발생했습니다.",
        ) from exc

    if not fulltext:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="본문을 추출할 수 없습니다.",
        )

    try:
        result = await ai_service(fulltext)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("AI 분류 중 오류 발생: %s", url)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI 처리 과정에서 오류가 발생했습니다.",
        ) from exc

    return result
