from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pydantic import BaseModel
from main import main

app = FastAPI(
    title="Speckle Server",
    version="1.0",
    description="An API server to answer questions regarding the Speckle Developer Docs"
)

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: dict

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Initialize the graph
graph = main()

# Add routes for the graph
add_routes(
    app,
    graph.with_types(input_type=Input, output_type=Output),
    path="/speckle_chat",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)