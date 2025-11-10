from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 기본 설정
    user_agent: str = "AI-Smell-Bot/0.1"
    article_timeout_seconds: int = 8
    ai_provider: str = Field(default="openai")
    temperature: float = Field(default=0.0, description="AI temperature")

    # OpenAI 설정
    openai_api_key: str
    openai_model: str = Field(default="gpt-4o-mini", description="기본 OpenAI 모델")

    # Gemini 설정
    gemini_api_key: str
    gemini_model: str = Field(default="gemini-2.5-pro", description="기본 Gemini 모델")

    model_config = SettingsConfigDict(env_file=".env")

    def __init__(self, **values):
        super().__init__(**values)
        if self.openai_api_key == "":
            raise ValueError("OpenAI API Key가 존재하지 않습니다.")

        if self.gemini_api_key == "":
            raise ValueError("Gemini API Key가 존재하지 않습니다.")


settings = Settings()
