from pydantic import AnyHttpUrl, BaseModel


class URLInputReq(BaseModel):
    url: AnyHttpUrl


class AIProbabilityRes(BaseModel):
    score: float
    reason: str


class ScanErrorRes(BaseModel):
    detail: str


class AIModelRes(BaseModel):
    label: str
    cues: list[str]
    reason: str
