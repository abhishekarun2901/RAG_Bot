from app.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import HybridVectorStore


def run_member2_verification():
    print("=" * 70)
    print("MEMBER 2 VERIFICATION RUNNER: HYBRID VECTOR STORE AUDIT")
    print("=" * 70)

    # 1. Process documents matching Member 1's DocumentProcessor signature
    print(f"\n[Step 1] Processing documents from '{settings.DOCUMENTS_DIR.resolve()}'...")
    processor = DocumentProcessor()
    chunks = processor.process_all_documents(settings.DOCUMENTS_DIR)
    print(f"-> Total chunks processed by DocumentProcessor: {len(chunks)}")

    if not chunks:
        print("[!] No chunks found. Place PDFs in ./documents or run 'python generate_test_docs.py'.")
        return

    # 2. Index chunks into Hybrid Vector Store
    print("\n[Step 2] Indexing chunks into Qdrant & BM25 Store...")
    vector_store = HybridVectorStore()
    vector_store.index_chunks(chunks)

    # 3. Test Standard Hybrid Query
    query_1 = "What are the safety and temperature requirements?"
    print(f"\n[Step 3] Querying Hybrid Search: '{query_1}' (top_k=3, threshold=0.35)")
    results_1 = vector_store.search(query=query_1, top_k=3, similarity_threshold=0.35)
    
    print(f"-> Retrieved {len(results_1)} relevant chunks:")
    for idx, res in enumerate(results_1, 1):
        print(f"   [{idx}] Document: {res['doc_name']} | Page: {res['page_number']} | Year: {res.get('year')} | Type: {res.get('doc_type')}")
        print(f"       Score: {res.get('similarity_score')} | RRF: {res.get('hybrid_rrf_score')}")
        print(f"       Text: {res['chunk_text'][:120]}...\n")

    # 4. Test Metadata Filter (doc_type attribute parsed by Member 1)
    sample_doc_type = chunks[0].get("doc_type")
    if sample_doc_type:
        print(f"[Step 4] Testing Metadata Filter: doc_type='{sample_doc_type}'")
        results_filter = vector_store.search(
            query="battery requirements",
            top_k=3,
            metadata_filter={"doc_type": sample_doc_type}
        )
        print(f"-> Filtered Results Count: {len(results_filter)}")
        for res in results_filter:
            print(f"   - Match: {res['doc_name']} (Type: {res['doc_type']}, Page {res['page_number']})")
            assert res["doc_type"] == sample_doc_type, "Metadata filter failed!"

    # 5. Test High Similarity Threshold (Hallucination Control / Out of Domain)
    query_irrelevant = "What is the orbital speed of Jupiter in kilometers per second?"
    print(f"\n[Step 5] Querying Out-Of-Domain Query: '{query_irrelevant}' (threshold=0.75)")
    results_out = vector_store.search(query=query_irrelevant, top_k=3, similarity_threshold=0.75)
    print(f"-> Chunks passing threshold 0.75: {len(results_out)}")
    assert len(results_out) == 0, "Threshold filtering failed! Irrelevant results were returned."

    print("\n" + "=" * 70)
    print("ALL MEMBER 2 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_member2_verification()