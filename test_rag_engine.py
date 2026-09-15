from app.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import HybridVectorStore
from app.services.llm_service import LLMService
from app.pipeline.rag_engine import RAGEngine


def run_member3_verification():
    print("=" * 70)
    print("MEMBER 3 VERIFICATION RUNNER: RAG ENGINE & GROQ LLM AUDIT")
    print("=" * 70)

    # 1. Verify Groq API Key
    if not settings.GROQ_API_KEY:
        print("[!] ERROR: GROQ_API_KEY is not configured in .env or settings.")
        return

    # 2. Ingest and Index Documents
    print(f"\n[Step 1] Ingesting documents from '{settings.DOCUMENTS_DIR.resolve()}'...")
    processor = DocumentProcessor()
    chunks = processor.process_all_documents(settings.DOCUMENTS_DIR)
    
    if not chunks:
        print("[!] No chunks found. Make sure PDFs exist in ./documents.")
        return

    print("\n[Step 2] Indexing chunks into Vector Store...")
    vector_store = HybridVectorStore()
    vector_store.index_chunks(chunks)

    # 3. Initialize Member 3 Components
    print("\n[Step 3] Initializing LLMService and RAGEngine...")
    llm_service = LLMService()
    rag_engine = RAGEngine(vector_store=vector_store, llm_service=llm_service)

    # 4. Test In-Domain Technical Query
    query_valid = "What are the temperature and thermal safety requirements?"
    print(f"\n[Step 4] Querying RAG Engine: '{query_valid}'")
    res_1 = rag_engine.query(user_query=query_valid, top_k=3, similarity_threshold=0.35)

    print(f"-> Context Retrieved: {res_1['context_retrieved']}")
    print(f"-> Generated Answer:\n{res_1['answer']}\n")
    print("-> Sources:")
    for src in res_1["sources"]:
        print(f"   - {src['doc_name']} (Page {src['page_number']})")

    assert res_1["context_retrieved"] is True, "In-domain query failed to retrieve context."
    assert len(res_1["sources"]) > 0, "No source citations returned for valid query."

    # 5. Test Out-of-Domain Query (Fallback Check)
    query_ood = "What is the orbital speed of Jupiter?"
    print(f"\n[Step 5] Querying Out-of-Domain Question: '{query_ood}'")
    res_2 = rag_engine.query(user_query=query_ood, similarity_threshold=0.75)

    print(f"-> Context Retrieved: {res_2['context_retrieved']}")
    print(f"-> Generated Answer: {res_2['answer']}")

    assert res_2["context_retrieved"] is False, "Out-of-domain query improperly retrieved context."
    assert res_2["answer"] == "I cannot answer this question based on the provided documents.", "Fallback guardrail failed."
    assert len(res_2["sources"]) == 0, "Out-of-domain response should have zero sources."

    print("\n" + "=" * 70)
    print("ALL MEMBER 3 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_member3_verification()