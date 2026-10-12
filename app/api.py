from pathlib import Path
from typing import Any
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from langserve import add_routes
from langchain_core.runnables import RunnableLambda

from app.graph import ask
from app.config import DOCS_DIR


app = FastAPI(
    title="AI Document Assistant API",
    description="RAG-based AI Document Assistant",
    version="1.0.0",
)

SUPPORTED_FILES = {".pdf", ".txt", ".docx"}


# ---------- Request Model -----------

class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default = 4, ge = 1, le = 20)


# ---------- Response Model ----------

class AskResponse(BaseModel):
    answer: str
    grounded: bool
    grounding_score: float
    grounding_reason: str
    sources: list[dict[str, Any]] = Field(default_factory = list)


# ---------- Common RAG Function ----------

def invoke_rag_graph(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question", "")).strip()
    top_k = int(payload.get("top_k", 4))

    if not question:
        raise ValueError("Question cannot be empty.")

    return ask(question, top_k)


# ---------- LangServe Route ----------

add_routes(app, RunnableLambda(invoke_rag_graph), path = "/ask")


# ---------- Home Route ----------

@app.get("/")
def home():
    return {
        "message": "AI Document Assistant API",
        "status": "running"
    }


# ---------- Ask Questions ----------

@app.post("/api/ask", response_model = AskResponse)
def ask_document(request: AskRequest):
    return invoke_rag_graph(request.model_dump())


# ---------- Upload Document ----------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file selected.")

    filename = Path(file.filename).name

    if Path(filename).suffix.lower() not in SUPPORTED_FILES:
        raise HTTPException(
            400, "Only PDF, DOCX, and TXT files are supported."
        )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCS_DIR / filename

    if destination.exists():
        raise HTTPException(409, "A document with this filename already exists.")

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from app.rag import rebuild_index
        result = rebuild_index()

        return {
            "message": "Document uploaded and indexed successfully.",
            "filename": filename,
            "documents": result["documents"],
            "chunks": result["chunks"],
        }

    except Exception:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            500, "Document upload or indexing failed."
        )

    finally:
        await file.close()


# ---------- Delete Document ----------

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    safe_filename = Path(filename).name
    file_path = DOCS_DIR / safe_filename

    if safe_filename != filename or not file_path.is_file():
        raise HTTPException(404, "Document not found.")

    if file_path.suffix.lower() not in SUPPORTED_FILES:
        raise HTTPException(400, "Unsupported document type.")

    try:
        file_path.unlink()

        from app.rag import rebuild_index
        result = rebuild_index()

        return {
            "message": "Document deleted and knowledge base updated.",
            "filename": safe_filename,
            "documents_indexed": result["documents"],
            "chunks_created": result["chunks"],
            "status": "updated",
        }

    except Exception:
        raise HTTPException(
            500, "Document deletion or index rebuilding failed."
        )