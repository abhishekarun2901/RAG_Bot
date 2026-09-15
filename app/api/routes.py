from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, IngestResponse
from app.services.document_processor import DocumentProcessor
from app.pipeline.graph import rag_graph
from app.config import settings
from app.services.vector_store import shared_vector_store as vector_store

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_documents():
    try:
        processor = DocumentProcessor()
        chunks = processor.process_all_documents(settings.DOCUMENTS_DIR)
        if not chunks:
            return IngestResponse(status="No documents found to ingest", total_chunks_indexed=0)
        vector_store.index_chunks(chunks)
        return IngestResponse(status="Success", total_chunks_indexed=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failure: {str(e)}")


@router.post("/chat", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    try:
        initial_state = {
            "user_query": request.query,
            "top_k": request.top_k or settings.DEFAULT_TOP_K,
            "similarity_threshold": request.similarity_threshold or settings.DEFAULT_SIMILARITY_THRESHOLD,
            "metadata_filter": request.metadata_filter,
            "effective_filter": None,
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "context_retrieved": False,
            "needs_expansion": False,
            "expansion_target": None,
            "expansion_attempts": 0,
            "expanded_contexts": []
        }

        final_state = rag_graph.invoke(initial_state)

        return QueryResponse(
            query=request.query,
            answer=final_state["answer"],
            sources=final_state["sources"],
            context_retrieved=final_state["context_retrieved"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failure: {str(e)}")