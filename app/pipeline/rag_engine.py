import re
from typing import Dict, Any, List, Optional
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


class RAGEngine:
    def __init__(self, vector_store: HybridVectorStore, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service

    def _extract_metadata_filters(self, query: str) -> Dict[str, Any]:
        """Extracts category metadata while treating numbers/years as semantic search terms."""
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

        return auto_filters

    def query(
        self,
        user_query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes grounded RAG pipeline:
        Stage 1: Metadata + Hybrid Search
        Stage 2: Full Corpus Hybrid Backfill (if Stage 1 < top_k)
        Stage 3: Relaxed Threshold Search
        Stage 4: Post-processed Grounded Synthesis & Clean Citations
        """
        auto_detected = self._extract_metadata_filters(user_query)
        combined_filters = {**auto_detected, **(metadata_filter or {})}
        effective_filter = combined_filters if combined_filters else None

        retrieved_chunks = []
        filter_used = None

        # --- STAGE 1: Target Search with Metadata Filter ---
        if effective_filter:
            retrieved_chunks = self.vector_store.search(
                query=user_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                metadata_filter=effective_filter
            )
            filter_used = effective_filter

        # --- STAGE 2: Hybrid Backfill across Full Corpus if context is thin ---
        if len(retrieved_chunks) < top_k:
            unfiltered_chunks = self.vector_store.search(
                query=user_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                metadata_filter=None
            )
            existing_ids = {c["chunk_id"] for c in retrieved_chunks}
            for chunk in unfiltered_chunks:
                if chunk["chunk_id"] not in existing_ids and len(retrieved_chunks) < top_k:
                    retrieved_chunks.append(chunk)
                    existing_ids.add(chunk["chunk_id"])

        # --- STAGE 3: Broad Semantic Fallback (Relaxed Similarity) ---
        if not retrieved_chunks and similarity_threshold > 0.30:
            retrieved_chunks = self.vector_store.search(
                query=user_query,
                top_k=top_k,
                similarity_threshold=0.30,
                metadata_filter=None
            )
            filter_used = None

        # Zero Chunks Retained -> Immediate Fallback
        if not retrieved_chunks:
            return {
                "query": user_query,
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "context_retrieved": False,
                "applied_filters": None
            }

        # --- Context Construction ---
        context_str = ""
        sources = []
        seen_sources = set()

        for idx, chunk in enumerate(retrieved_chunks, 1):
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

        formatted_user_prompt = f"Retrieved Context:\n{context_str}\n\nUser Question: {user_query}"

        # --- LLM Synthesis ---
        raw_answer = self.llm_service.generate_answer(
            prompt=formatted_user_prompt,
            system_prompt=SYSTEM_PROMPT
        )

        clean_answer = raw_answer.strip()

        # --- STAGE 4: Post-Processing & Hallucinated Citation Guardrail ---
        if FALLBACK_RESPONSE.lower() in clean_answer.lower():
            return {
                "query": user_query,
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "context_retrieved": False,
                "applied_filters": filter_used
            }

        return {
            "query": user_query,
            "answer": clean_answer,
            "sources": sources,
            "context_retrieved": True,
            "applied_filters": filter_used
        }