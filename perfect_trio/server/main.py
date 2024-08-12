from fastapi import FastAPI
from starlette.responses import RedirectResponse  # Starlette에서 가져오기
from models import Input, Output  # server/models.py에서 정의된 Pydantic 모델 임포트
import uvicorn
from langserve import add_routes

# FastAPI 앱 설정
app = FastAPI(
    title="Speckle Server",
    version="1.0",
    description="An API server to answer questions regarding the Speckle Developer Docs"
)

# 루트 URL 리디렉션
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# API 경로 추가
add_routes(
    app,
    chain.with_types(input_type=Input, output_type=Output),
    path="/speckle_chat",
)

# 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
