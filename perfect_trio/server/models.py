from pydantic import BaseModel

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: dict
