from fastapi import FastAPI

from app.api.v1.endpoints.scan import router

app = FastAPI(title="AI Smell")

app.include_router(router, prefix="/api/v1")
