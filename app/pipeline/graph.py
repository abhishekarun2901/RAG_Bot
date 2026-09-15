import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from app.services.vector_store import HybridVectorStore
from app.services.llm_service import LLMService
from app.config import settings

FALLBACK_RESPONSE = "I cannot answer this question based on the provided documents."

SYSTEM_PROMPT = """You are an expert technical AI assistant specializing in enterprise documentation and compliance regulations.

INSTRUCTIONS:
1. Answer the user's question clearly, directly, and professionally based STRICTLY on the provided retrieved context.
2. Format your response cleanly using standard Markdown (bolding, clean lists). Do NOT include conversational setup or fluff (e.g., "Here is the summary").
3. Cite sources inline at the end of relevant facts using this exact format: [Doc Name, Page X].
4. IF AND ONLY IF the context lacks sufficient information to answer the question, state EXACTLY:
   "I cannot answer this question based on the provided documents."
   Do NOT attach any citations, quotes, or source references if the context is insufficient.
"""


class RAGState(TypedDict):
    user_query: str
    top_k: int
    similarity_threshold: float
    metadata_filter: Optional[Dict[str, Any]]
    effective_filter: Optional[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    context_retrieved: bool


# Global service instances for graph execution
vector_store = HybridVectorStore()
llm_service = None


def get_llm_service() -> LLMService:
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service


# --- Node Definitions ---

def extract_metadata_node(state: RAGState) -> Dict[str, Any]:
    query = state["user_query"]
    auto_filters = {}
    query_lower = query.lower()

    if "r100" in query_lower or "ece r100" in query_lower:
        auto_filters["doc_type"] = "ECE_REGULATION"
    elif "iso 6469" in query_lower or "iso standard" in query_lower:
        auto_filters["doc_type"] = "ISO_STANDARD"
    elif "un 383" in query_lower or "un383" in query_lower:
        auto_filters["doc_type"] = "UN_TRANSPORT"
    elif "fmvss" in query_lower or "nhtsa" in query_lower:
        auto_filters["doc_type"] = "NHTSA_FMVSS"
    elif "eu battery directive" in query_lower or "directive 2024" in query_lower:
        auto_filters["doc_type"] = "EU_DIRECTIVE"

    explicit = state.get("metadata_filter") or {}
    combined = {**auto_filters, **explicit}
    return {"effective_filter": combined if combined else None}


def targeted_search_node(state: RAGState) -> Dict[str, Any]:
    effective_filter = state.get("effective_filter")
    if not effective_filter:
        return {"retrieved_chunks": []}

    chunks = vector_store.search(
        query=state["user_query"],
        top_k=state["top_k"],
        similarity_threshold=state["similarity_threshold"],
        metadata_filter=effective_filter
    )
    return {"retrieved_chunks": chunks}


def backfill_search_node(state: RAGState) -> Dict[str, Any]:
    retrieved_chunks = list(state.get("retrieved_chunks") or [])
    top_k = state["top_k"]

    unfiltered_chunks = vector_store.search(
        query=state["user_query"],
        top_k=top_k,
        similarity_threshold=state["similarity_threshold"],
        metadata_filter=None
    )

    existing_ids = {c["chunk_id"] for c in retrieved_chunks}
    for chunk in unfiltered_chunks:
        if chunk["chunk_id"] not in existing_ids and len(retrieved_chunks) < top_k:
            retrieved_chunks.append(chunk)
            existing_ids.add(chunk["chunk_id"])

    return {"retrieved_chunks": retrieved_chunks}


def relaxed_search_node(state: RAGState) -> Dict[str, Any]:
    chunks = vector_store.search(
        query=state["user_query"],
        top_k=state["top_k"],
        similarity_threshold=0.30,
        metadata_filter=None
    )
    return {"retrieved_chunks": chunks, "effective_filter": None}


def synthesize_node(state: RAGState) -> Dict[str, Any]:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {
            "answer": FALLBACK_RESPONSE,
            "sources": [],
            "context_retrieved": False
        }

    context_str = ""
    sources = []
    seen_sources = set()

    for idx, chunk in enumerate(chunks, 1):
        context_str += f"\n--- Context Block {idx} ---\n"
        context_str += f"Document: {chunk['doc_name']} (Page {chunk['page_number']})\n"
        context_str += f"Content: {chunk['chunk_text']}\n"

        source_key = (chunk["doc_name"], chunk["page_number"])
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "doc_name": chunk["doc_name"],
                "page_number": chunk["page_number"],
                "score": chunk.get("similarity_score") or chunk.get("hybrid_rrf_score")
            })

    formatted_user_prompt = f"Retrieved Context:\n{context_str}\n\nUser Question: {state['user_query']}"
    service = get_llm_service()

    raw_answer = service.generate_answer(
        prompt=formatted_user_prompt,
        system_prompt=SYSTEM_PROMPT
    )
    clean_answer = raw_answer.strip()

    if FALLBACK_RESPONSE.lower() in clean_answer.lower():
        return {
            "answer": FALLBACK_RESPONSE,
            "sources": [],
            "context_retrieved": False
        }

    return {
        "answer": clean_answer,
        "sources": sources,
        "context_retrieved": True
    }


# --- Conditional Routing Logic ---

def route_after_targeted(state: RAGState) -> str:
    if len(state.get("retrieved_chunks", [])) < state["top_k"]:
        return "backfill_search"
    return "synthesize"


def route_after_backfill(state: RAGState) -> str:
    if not state.get("retrieved_chunks") and state["similarity_threshold"] > 0.30:
        return "relaxed_search"
    return "synthesize"


# --- Graph Construction ---

builder = StateGraph(RAGState)

builder.add_node("extract_metadata", extract_metadata_node)
builder.add_node("targeted_search", targeted_search_node)
builder.add_node("backfill_search", backfill_search_node)
builder.add_node("relaxed_search", relaxed_search_node)
builder.add_node("synthesize", synthesize_node)

builder.set_entry_point("extract_metadata")
builder.add_edge("extract_metadata", "targeted_search")
builder.add_conditional_edges("targeted_search", route_after_targeted, {
    "backfill_search": "backfill_search",
    "synthesize": "synthesize"
})
builder.add_conditional_edges("backfill_search", route_after_backfill, {
    "relaxed_search": "relaxed_search",
    "synthesize": "synthesize"
})
builder.add_edge("relaxed_search", "synthesize")
builder.add_edge("synthesize", END)

rag_graph = builder.compile()