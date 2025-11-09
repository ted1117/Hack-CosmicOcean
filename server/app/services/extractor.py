import logging

import aiohttp
import trafilatura

from app.core.config import settings

logger = logging.getLogger(__name__)


async def download_html(url: str) -> None | str:
    try:
        timeout = aiohttp.ClientTimeout(total=settings.article_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(
                url,
                headers={"User-Agent": settings.user_agent},
                allow_redirects=True,
            ) as r:
                if r.status >= 400:
                    logger.warning(f"HTTP {r.status} 오류: {url}")
                    return None
                html = await r.text(errors="ignore")
                return html
    except (aiohttp.ClientError, TimeoutError) as e:
        logger.error(f"URL 크롤링 실패: {url}, 오류: {e}")
        return None
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {url}, 오류: {e}", exc_info=True)
        return None


def extract_text_from_html(html: str, url: str) -> None | str:
    if not html:
        logger.warning("HTML이 비어있습니다.")
        return None

    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        return text
    except Exception as e:
        logger.error(f"본문 추출 실패: {url}, 오류: {e}")
        return None


async def extract_article(url: str) -> None | str:
    html = await download_html(url)
    if not html:
        return None

    return extract_text_from_html(html, url)
