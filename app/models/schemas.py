from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
 
 
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="User question (3 to 500 characters)"
    )
    top_k: Optional[int] = Field(default=4, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(default=0.45, ge=0.0, le=1.0)
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Key-value payload filter")
 
 
class SourceMetadata(BaseModel):
    doc_name: str
    page_number: int
    score: Optional[float] = None
 
 
class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceMetadata]
    context_retrieved: bool
 
 
class IngestResponse(BaseModel):
    status: str
    total_chunks_indexed: int