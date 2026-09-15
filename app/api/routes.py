from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, IngestResponse
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import HybridVectorStore
from app.services.llm_service import LLMService
from app.pipeline.rag_engine import RAGEngine
from app.config import settings
 
router = APIRouter()
 
# Global pipeline instances
vector_store = HybridVectorStore()
llm_service = None
rag_engine = None
 
 
def get_rag_engine() -> RAGEngine:
    global llm_service, rag_engine
    if rag_engine is None:
        llm_service = LLMService()
        rag_engine = RAGEngine(vector_store=vector_store, llm_service=llm_service)
    return rag_engine
 
 
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
        engine = get_rag_engine()
        result = engine.query(
            user_query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            metadata_filter=request.metadata_filter
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")