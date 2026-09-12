import sys
import json
from pathlib import Path

# Force unbuffered output so Windows PowerShell flushes logs immediately
sys.stdout.reconfigure(encoding="utf-8")

print(">>> Starting ingest_documents.py runner...", flush=True)

try:
    from app.config import settings
    from app.services.document_processor import DocumentProcessor
    print(">>> Imports resolved successfully.", flush=True)
except Exception as err:
    print(f"[CRITICAL IMPORT ERROR]: {err}", flush=True)
    sys.exit(1)


def run_ingestion_verification():
    print("=" * 70, flush=True)
    print("MEMBER 1: DOCUMENT INGESTION & SEMANTIC CHUNKING VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    docs_dir = Path(settings.DOCUMENTS_DIR)
    print(f"Target documents directory: {docs_dir.resolve()}", flush=True)

    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        print(f"[!] Created missing directory: {docs_dir.resolve()}", flush=True)
        print("Please place at least 10 PDF documents into this folder and re-run.", flush=True)
        return

    pdf_files = list(docs_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF documents in '{docs_dir}'.", flush=True)

    if len(pdf_files) == 0:
        print("[!] No PDF documents found. Run 'python generate_test_docs.py' first.", flush=True)
        return

    if len(pdf_files) < 10:
        print(f"[!] Warning: Project specification requests >= 10 documents. Current count: {len(pdf_files)}", flush=True)

    print("\n[1/3] Initializing DocumentProcessor & Loading SentenceTransformer...", flush=True)
    processor = DocumentProcessor()

    print("\n[2/3] Executing Extraction, Sanitization, & Semantic Chunking...", flush=True)
    all_chunks = processor.process_all_documents(docs_dir)

    print(f"\n[3/3] Ingestion complete. Generated {len(all_chunks)} semantic chunks.", flush=True)

    if not all_chunks:
        print("[ERROR] No chunks generated. Verify that PDFs have readable text content.", flush=True)
        return

    print("\n" + "-" * 70, flush=True)
    print("DATA INTEGRITY AUDIT", flush=True)
    print("-" * 70, flush=True)

    sample = all_chunks[0]
    required_keys = ["chunk_id", "doc_id", "doc_name", "page_number", "section", "chunk_text", "metadata"]
    missing_keys = [k for k in required_keys if k not in sample]

    if missing_keys:
        print(f"[FAIL] Schema mismatch. Missing keys: {missing_keys}", flush=True)
    else:
        print("[PASS] Schema validated. All required metadata fields are present.", flush=True)

    print("\nSAMPLE CHUNK PAYLOAD (Delivered to Member 2 / Vector Store):", flush=True)
    print(json.dumps(sample, indent=2), flush=True)

    total_chars = sum(len(c["chunk_text"]) for c in all_chunks)
    avg_chars = total_chars // len(all_chunks)
    print("\n" + "-" * 70, flush=True)
    print("PIPELINE SUMMARY METRICS:", flush=True)
    print(f"- Total Documents Processed: {len(pdf_files)}", flush=True)
    print(f"- Total Semantic Chunks:     {len(all_chunks)}", flush=True)
    print(f"- Average Chunk Length:      {avg_chars} characters", flush=True)
    print("-" * 70, flush=True)


if __name__ == "__main__":
    run_ingestion_verification()