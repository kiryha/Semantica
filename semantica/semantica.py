from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

import core

app = FastAPI()

class ComputeRequest(BaseModel):
    expression: str

@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")

@app.post("/compute")
def compute(req: ComputeRequest):
    return {"result": core.solve(req.expression, core.embeddings) or ""}
