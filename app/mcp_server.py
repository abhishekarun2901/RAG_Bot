import sys
from pathlib import Path
from typing import Optional

import pymupdf
from fastmcp import FastMCP

from app.config import settings
from app.services.vector_store import shared_vector_store as vector_store

mcp = FastMCP("Enterprise-RAG-Document-Server")


@mcp.tool()
def search_enterprise_documents(
    query: str,
    doc_type: Optional[str] = None,
    year: Optional[int] = None,
    top_k: int = settings.DEFAULT_TOP_K,
) -> str:
    """Search indexed compliance regulations, ISO standards, and safety technical documentation.

    Args:
        query: Semantic or lexical query string.
        doc_type: Optional category filter (e.g. 'ECE_REGULATION', 'ISO_STANDARD',
          'UN_TRANSPORT', 'EU_DIRECTIVE').
        year: Optional 4-digit publishing year constraint.
        top_k: Maximum number of document chunks to retrieve (1 to 20).
    """
    metadata_filter = {}
    if doc_type:
        metadata_filter["doc_type"] = doc_type
    if year:
        metadata_filter["year"] = year

    results = vector_store.search(
        query=query,
        top_k=top_k,
        similarity_threshold=settings.DEFAULT_SIMILARITY_THRESHOLD,
        metadata_filter=metadata_filter if metadata_filter else None,
    )

    if not results:
        return "No relevant context found matching the criteria."

    formatted_output = []
    for idx, chunk in enumerate(results, 1):
        formatted_output.append(
            f"--- Document Block {idx}: {chunk['doc_name']} (Page {chunk['page_number']}) ---\n"
            f"{chunk['chunk_text']}\n"
        )

    return "\n".join(formatted_output)


@mcp.tool()
def fetch_document_page_context(
    doc_name: str, page_number: int, window_size: int = 0
) -> str:
    """Retrieves full sanitized text of a specific page and optionally its adjacent pages

    from a document. Used by Self-RAG loops when a retrieved chunk indicates
    missing
    context, tables, or subsequent regulatory clauses.

    Args:
        doc_name: Filename of the target PDF (e.g.,
          'iso_6469_part1_safety_2025.pdf').
        page_number: Target page number (1-indexed).
        window_size: Number of adjacent pages before and after to include (0 to
          2).
    """
    file_path = settings.DOCUMENTS_DIR / doc_name
    if not file_path.exists():
        return f"Error: Document '{doc_name}' not found in registry."

    try:
        with pymupdf.open(file_path) as doc:
            total_pages = len(doc)
            start_page = max(1, page_number - window_size)
            end_page = min(total_pages, page_number + window_size)

            extracted_sections = []
            for p_idx in range(start_page - 1, end_page):
                page = doc[p_idx]
                blocks = page.get_text("blocks")
                text_segments = [
                    b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()
                ]
                full_page_text = "\n".join(text_segments)

                extracted_sections.append(
                    f"=== {doc_name} | PAGE {p_idx + 1} OF {total_pages} ===\n{full_page_text}"
                )

            return "\n\n".join(extracted_sections)

    except Exception as err:
        return f"Error reading document: {str(err)}"


if __name__ == "__main__":
    mcp.run()