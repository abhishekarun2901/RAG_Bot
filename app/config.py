from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Directory paths
    DOCUMENTS_DIR: Path = Path("./documents")
    SUPPORTED_EXTENSIONS: List[str] = [".pdf"]

    # Semantic chunking configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Statistical percentile cut-off for cosine distance spike (0.0 to 100.0)
    BREAKPOINT_PERCENTILE_THRESHOLD: float = 85.0

    # Sentence context window buffer (number of adjacent sentences to merge for context)
    BUFFER_SIZE: int = 1

    # Hard bounds to prevent microscopic chunks or LLM context window blowups
    MIN_CHUNK_SIZE: int = 50
    MAX_CHUNK_SIZE: int = 1200

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
