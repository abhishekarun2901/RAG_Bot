from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Member 1 Settings: Ingestion & Semantic Chunking
    DOCUMENTS_DIR: Path = Path("./documents")
    SUPPORTED_EXTENSIONS: List[str] = [".pdf"]
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    BREAKPOINT_PERCENTILE_THRESHOLD: float = 85.0
    BUFFER_SIZE: int = 1
    MIN_CHUNK_SIZE: int = 50
    MAX_CHUNK_SIZE: int = 1200

    # Member 2 Settings: Qdrant Vector DB & Hybrid Search
    QDRANT_STORAGE_PATH: Path = Path("./qdrant_data")
    QDRANT_COLLECTION_NAME: str = "rag_documents"
    DEFAULT_TOP_K: int = 4
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.45
    HYBRID_DENSE_WEIGHT: float = 0.7
    HYBRID_SPARSE_WEIGHT: float = 0.3

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()