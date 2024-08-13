import sys
import os

# utils 폴더를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

from fastapi import FastAPI
from starlette.responses import RedirectResponse
from models import Input, Output
from graph_setup import create_chain
import uvicorn
from langserve import add_routes
import dotenv # pip install python-dotenv
from dotenv import load_dotenv, find_dotenv

# 최신 LangChain 사용
from langchain_openai import OpenAI  # 수정된 부분
from langchain_community.vectorstores import FAISS
from grader_utils import GraderUtils
from document_loader import DocumentLoader


# .env 파일에서 환경 변수 불러오기
load_dotenv(r'C:\Users\user\Desktop\LangChain\Daily\all.env') 

# 환경 변수에 API 키 설정
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# Langsmith Tracing 설정
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Fire Crawl API 설정
os.environ['FIRE_API_KEY'] = os.getenv('FIRE_API_KEY')


# LLM 초기화 (예: OpenAI API 사용)
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 문서 로더 및 벡터 스토어 초기화 (예시 코드)
loader = DocumentLoader(api_key=os.getenv("FIRE_API_KEY"))
docs = loader.get_docs("https://speckle.guide/dev/server-graphql-api.html#advanced-queries")
retriever = FAISS.from_documents(docs, llm)

# GraderUtils 객체 생성
grader_utils = GraderUtils(llm)
retrieval_grader = grader_utils.create_retrieval_grader()
hallucination_grader = grader_utils.create_hallucination_grader()
code_evaluator = grader_utils.create_code_evaluator()
question_rewriter = grader_utils.create_question_rewriter()

# 그래프 체인 생성
chain = create_chain(llm, retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter)

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
