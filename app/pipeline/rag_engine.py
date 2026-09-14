from typing import Dict, Any, List, Optional
from app.services.vector_store import HybridVectorStore
from app.services.llm_service import LLMService
from app.config import settings

SYSTEM_PROMPT = """You are an expert assistant for enterprise technical documentation and compliance regulations.
Answer the user's question STRICTLY based on the provided retrieved document context below.

Rules:
1. If the context does not contain sufficient information to answer the question, state clearly: "I cannot answer this question based on the provided documents."
2. Do not use external knowledge or fabricate statements not directly supported by the context.
3. Cite your sources directly in your response using format: [Doc Name, Page X].
"""


class RAGEngine:
    def __init__(self, vector_store: HybridVectorStore, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    def query(
        self,
        user_query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Runs end-to-end RAG pipeline: threshold search -> groundedness guardrail -> LLM synthesis."""
        # 1. Hybrid Search via Member 2's Store
        retrieved_chunks = self.vector_store.search(
            query=user_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter
        )

        # 2. Hallucination Control: Zero context returned -> Fallback
        if not retrieved_chunks:
            return {
                "query": user_query,
                "answer": "I cannot answer this question based on the provided documents.",
                "sources": [],
                "context_retrieved": False
            }

        # 3. Context Construction
        context_str = ""
        sources = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            context_str += f"\n--- Context Block {idx} ---\n"
            context_str += f"Document: {chunk['doc_name']} (Page {chunk['page_number']})\n"
            context_str += f"Content: {chunk['chunk_text']}\n"
            
            sources.append({
                "doc_name": chunk["doc_name"],
                "page_number": chunk["page_number"],
                "score": chunk.get("similarity_score") or chunk.get("hybrid_rrf_score")
            })

        formatted_user_prompt = f"Retrieved Context:\n{context_str}\n\nUser Question: {user_query}"

        # 4. LLM Response Generation
        answer = self.llm_service.generate_answer(
            prompt=formatted_user_prompt,
            system_prompt=SYSTEM_PROMPT
        )

        return {
            "query": user_query,
            "answer": answer,
            "sources": sources,
            "context_retrieved": True
        }