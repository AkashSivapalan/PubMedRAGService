import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from ingest import ingest, ingest_all_topics, MEDICAL_TOPICS
from retriever import HybridRetriever
from chain import PubMedRAGChain
from session import create_session_id, get_history, save_history, delete_history, session_exists

load_dotenv()

CHROMA_PATH = "./chroma_db"
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

app_state = {}


def load_from_chroma() -> list[Document]:
    """Reconstruct chunks from existing ChromaDB for BM25 index."""
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=EMBEDDINGS,
        collection_name="pubmed"
    )
    data = vectorstore.get()
    return [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(data["documents"], data["metadatas"])
    ]


def build_rag(chunks: list[Document]) -> PubMedRAGChain:
    retriever = HybridRetriever(chunks, persist_dir=CHROMA_PATH)
    return PubMedRAGChain(retriever, model="gpt-4o")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("Found existing ChromaDB — loading without re-ingesting...")
        chunks = load_from_chroma()
        print(f"Loaded {len(chunks)} chunks from ChromaDB.")
    else:
        print("No ChromaDB found — building broad medical knowledge base...")
        print(f"Ingesting {len(MEDICAL_TOPICS)} topics. This will take several minutes...")
        chunks = ingest_all_topics()

    app_state["chunks"] = chunks
    app_state["rag"] = build_rag(chunks)
    print("Ready.")
    yield
    app_state.clear()


app = FastAPI(
    title="PubMed RAG API",
    description="Query broad medical literature using RAG",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in production
    allow_methods=["*"],
    allow_headers=["*"]
)


# --- Request / Response models ---

class QueryRequest(BaseModel):
    question: str
    session_id: str
    model: str = "gpt-4o"


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class IngestRequest(BaseModel):
    """
    Ingest a single new topic and append it to the existing knowledge base.
    Set replace=True to wipe the existing KB and start fresh with this topic only.
    """
    topic: str
    max_results: int = 200
    replace: bool = False


class IngestAllRequest(BaseModel):
    """Re-ingest the full default MEDICAL_TOPICS list from scratch."""
    confirm: bool = False  # safety flag — must be True to proceed


class ResetRequest(BaseModel):
    session_id: str


# --- Routes ---

@app.get("/health")
def health():
    return {
        "status": "ok",
        "kb_loaded": "rag" in app_state,
        "chunk_count": len(app_state.get("chunks", []))
    }


@app.get("/topics")
def list_topics():
    """List the default medical topics that will be ingested."""
    return {
        "topics": [
            {"query": q, "max_results": n}
            for q, n in MEDICAL_TOPICS
        ]
    }


@app.post("/session")
def new_session():
    """Create a new session. Call this once when a conversation starts."""
    session_id = create_session_id()
    return {"session_id": session_id}


@app.get("/session/{session_id}")
def check_session(session_id: str):
    """Check whether a session ID is currently active."""
    active = session_exists(session_id)
    if not active:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return {"session_id": session_id, "active": True}


@app.post("/ingest")
def ingest_topic(request: IngestRequest):
    """
    Add a single topic to the knowledge base.
    Appends by default — set replace=True to wipe and start fresh.
    """
    try:
        if request.replace:
            # Wipe existing collection first
            old_store = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=EMBEDDINGS,
                collection_name="pubmed"
            )
            old_store.delete_collection()

        chunks = ingest(request.topic, request.max_results, replace=request.replace)

        if request.replace:
            app_state["chunks"] = chunks
        else:
            # Merge new chunks with existing for BM25
            existing = app_state.get("chunks", [])
            app_state["chunks"] = existing + chunks

        app_state["rag"] = build_rag(app_state["chunks"])
        return {
            "message": f"Ingested {len(chunks)} chunks for: '{request.topic}'",
            "total_chunks": len(app_state["chunks"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/all")
def ingest_all(request: IngestAllRequest):
    """
    Re-ingest the full MEDICAL_TOPICS list from scratch.
    Requires confirm=True to prevent accidental wipes.
    Warning: this takes 20-40 minutes depending on topic count.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to proceed. This wipes and rebuilds the entire knowledge base."
        )
    try:
        chunks = ingest_all_topics()
        app_state["chunks"] = chunks
        app_state["rag"] = build_rag(chunks)
        return {
            "message": f"Rebuilt full knowledge base.",
            "total_chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Query the knowledge base."""
    if "rag" not in app_state:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base not loaded. Call /ingest first."
        )
    try:
        history = get_history(request.session_id)
        result = app_state["rag"].query(request.question, history)

        history.append((request.question, result["answer"]))
        save_history(request.session_id, history)

        sources = [
            {
                "title": doc.metadata.get("title", "Unknown"),
                "journal": doc.metadata.get("journal", ""),
                "year": doc.metadata.get("year", ""),
                "url": doc.metadata.get("source", ""),
                "excerpt": doc.page_content[:300]
            }
            for doc in result["sources"]
        ]
        return QueryResponse(answer=result["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset_memory(request: ResetRequest):
    """Wipe conversation history for a session."""
    delete_history(request.session_id)
    return {"message": "Session reset."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
