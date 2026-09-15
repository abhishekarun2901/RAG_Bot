from typing import Optional
from fastmcp import FastMCP
from app.services.vector_store import HybridVectorStore
from app.config import settings

mcp = FastMCP("Enterprise-RAG-Document-Server")
vector_store = HybridVectorStore()


@mcp.tool()
def search_enterprise_documents(
    query: str,
    doc_type: Optional[str] = None,
    year: Optional[int] = None,
    top_k: int = settings.DEFAULT_TOP_K
) -> str:
    """
    Search indexed compliance regulations, ISO standards, and safety technical documentation.
    
    Args:
        query: Semantic or lexical query string.
        doc_type: Optional category filter (e.g. 'ECE_REGULATION', 'ISO_STANDARD', 'UN_TRANSPORT', 'EU_DIRECTIVE').
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
        metadata_filter=metadata_filter if metadata_filter else None
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


if __name__ == "__main__":
    mcp.run()