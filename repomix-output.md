This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
app/
  pipeline/
    rag_engine.py
  services/
    document_processor.py
    llm_service.py
    vector_store.py
  config.py
.gitignore
generate_test_docs.py
ingest_documents.py
requirements.txt
test_rag_engine.py
test_vector_store.py
```

# Files

## File: app/pipeline/rag_engine.py
```python
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
```

## File: app/services/document_processor.py
```python
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings


class DocumentProcessor:
    def __init__(
        self,
        embedding_model_name: str = settings.EMBEDDING_MODEL_NAME,
        percentile_threshold: float = settings.BREAKPOINT_PERCENTILE_THRESHOLD,
        buffer_size: int = settings.BUFFER_SIZE,
        min_chunk_size: int = settings.MIN_CHUNK_SIZE,
        max_chunk_size: int = settings.MAX_CHUNK_SIZE,
    ):
        self.percentile_threshold = percentile_threshold
        self.buffer_size = buffer_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.encoder = SentenceTransformer(embedding_model_name)

    def generate_document_id(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return f"doc_{hasher.hexdigest()[:12]}"

    def parse_document_attributes(self, filename: str, first_page_text: str) -> Dict[str, Any]:
        """Extracts contextual metadata (year, regulation type) for advanced filtering."""
        # 1. Extract 4-digit Year (e.g. 2021 to 2026)
        year_match = re.search(r"\b(202[0-9])\b", filename) or re.search(r"\b(202[0-9])\b", first_page_text)
        year = int(year_match.group(1)) if year_match else None

        # 2. Extract Document Category / Standard Type
        fn_lower = filename.lower()
        if "r100" in fn_lower:
            doc_type = "ECE_REGULATION"
        elif "iso" in fn_lower:
            doc_type = "ISO_STANDARD"
        elif "un383" in fn_lower:
            doc_type = "UN_TRANSPORT"
        elif "fmvss" in fn_lower or "nhtsa" in fn_lower:
            doc_type = "NHTSA_FMVSS"
        elif "eu_battery" in fn_lower:
            doc_type = "EU_DIRECTIVE"
        else:
            doc_type = "TECHNICAL_SPEC"

        return {
            "year": year,
            "doc_type": doc_type
        }

    def clean_text(self, text: str) -> str:
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text.strip()

    def extract_text_by_pages(self, file_path: Path) -> List[Dict[str, Any]]:
        doc_id = self.generate_document_id(file_path)
        doc_name = file_path.name
        extracted_pages = []

        with pymupdf.open(file_path) as doc:
            first_page_text = doc[0].get_text() if len(doc) > 0 else ""
            attributes = self.parse_document_attributes(doc_name, first_page_text)

            for page_index in range(len(doc)):
                page = doc[page_index]
                blocks = page.get_text("blocks")
                page_text_segments = []

                for b in blocks:
                    if b[6] == 0:
                        cleaned_block = self.clean_text(b[4])
                        if cleaned_block:
                            page_text_segments.append(cleaned_block)

                combined_page_text = " ".join(page_text_segments)

                if combined_page_text:
                    extracted_pages.append(
                        {
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "page_number": page_index + 1,
                            "text": combined_page_text,
                            "year": attributes["year"],
                            "doc_type": attributes["doc_type"]
                        }
                    )

        return extracted_pages

    def split_into_sentences(self, text: str) -> List[str]:
        abbreviations = [
            "e.g.", "i.e.", "Dr.", "Mr.", "Ms.", "Mrs.", "Prof.", "Inc.", 
            "Ltd.", "Corp.", "vs.", "v.", "etc.", "approx.", "dept.", "Rev.", "No."
        ]
        masked_text = text
        masks = {}
        for idx, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{idx}__"
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            if pattern.search(masked_text):
                masks[placeholder] = abbr
                masked_text = pattern.sub(placeholder, masked_text)

        decimals = re.findall(r"\b\d+\.\d+\b", masked_text)
        for idx, dec in enumerate(decimals):
            placeholder = f"__DEC{idx}__"
            masks[placeholder] = dec
            masked_text = masked_text.replace(dec, placeholder, 1)

        raw_sentences = re.split(r"(?<=[.?!])\s+", masked_text)

        restored_sentences = []
        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            for placeholder, original in masks.items():
                s_clean = s_clean.replace(placeholder, original)
            if len(s_clean) > 5:
                restored_sentences.append(s_clean)

        return restored_sentences

    def _create_buffered_sentences(self, sentences: List[str]) -> List[str]:
        buffered = []
        n = len(sentences)
        for i in range(n):
            start = max(0, i - self.buffer_size)
            end = min(n, i + self.buffer_size + 1)
            buffered.append(" ".join(sentences[start:end]))
        return buffered

    def semantic_chunk_text(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_text = page_data["text"]
        sentences = self.split_into_sentences(raw_text)

        if len(sentences) <= 1:
            if len(raw_text) >= self.min_chunk_size:
                return [self._format_chunk(page_data, raw_text, chunk_index=0)]
            return []

        buffered_sentences = self._create_buffered_sentences(sentences)
        embeddings = self.encoder.encode(buffered_sentences, show_progress_bar=False)

        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1.0 - float(np.clip(sim, 0.0, 1.0))
            distances.append(distance)

        breakpoint_threshold = float(
            np.percentile(distances, self.percentile_threshold)
        )

        chunks = []
        current_chunk_sentences = [sentences[0]]
        chunk_idx = 0

        for i, dist in enumerate(distances):
            sentence_to_add = sentences[i + 1]
            current_chunk_len = len(" ".join(current_chunk_sentences))

            is_semantic_shift = dist > breakpoint_threshold
            is_oversized = (current_chunk_len + len(sentence_to_add)) > self.max_chunk_size

            if (is_semantic_shift or is_oversized) and current_chunk_len >= self.min_chunk_size:
                chunk_str = " ".join(current_chunk_sentences)
                chunks.append(self._format_chunk(page_data, chunk_str, chunk_idx))
                current_chunk_sentences = [sentence_to_add]
                chunk_idx += 1
            else:
                current_chunk_sentences.append(sentence_to_add)

        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences)
            if len(chunk_str) >= self.min_chunk_size:
                chunks.append(self._format_chunk(page_data, chunk_str, chunk_idx))

        return chunks

    def _format_chunk(
        self, page_data: Dict[str, Any], chunk_text: str, chunk_index: int
    ) -> Dict[str, Any]:
        chunk_id = f"{page_data['doc_id']}_p{page_data['page_number']}_c{chunk_index}"
        return {
            "chunk_id": chunk_id,
            "doc_id": page_data["doc_id"],
            "doc_name": page_data["doc_name"],
            "page_number": page_data["page_number"],
            "section": f"Page {page_data['page_number']} - Segment {chunk_index + 1}",
            "chunk_text": chunk_text,
            "year": page_data.get("year"),
            "doc_type": page_data.get("doc_type"),
            "metadata": {
                "year": page_data.get("year"),
                "doc_type": page_data.get("doc_type"),
                "character_count": len(chunk_text),
                "estimated_token_count": len(chunk_text) // 4,
                "chunk_strategy": "semantic_percentile",
            },
        }

    def process_all_documents(self, documents_dir: Path = settings.DOCUMENTS_DIR) -> List[Dict[str, Any]]:
        if not documents_dir.exists():
            raise FileNotFoundError(f"Target directory '{documents_dir}' does not exist.")

        all_chunks = []
        pdf_paths = list(documents_dir.glob("*.pdf"))

        for pdf_path in sorted(pdf_paths):
            print(f"Processing: {pdf_path.name}")
            pages = self.extract_text_by_pages(pdf_path)
            for page_data in pages:
                page_chunks = self.semantic_chunk_text(page_data)
                all_chunks.extend(page_chunks)

        return all_chunks
```

## File: app/services/llm_service.py
```python
import os
from groq import Groq
from app.config import settings


class LLMService:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate_answer(self, prompt: str, system_prompt: str) -> str:
        """Calls Groq API with low temperature for strictly deterministic factual generation."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content
```

## File: app/services/vector_store.py
```python
import re
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.config import settings


class HybridVectorStore:
    def __init__(self):
        # 1. Initialize Dense Embedding Encoder
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self.embedding_dim = self.encoder.get_embedding_dimension()
        
        # 2. Initialize Embedded Local Qdrant Instance
        storage_path_str = str(settings.QDRANT_STORAGE_PATH.resolve())
        self.client = QdrantClient(path=storage_path_str)
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._init_collection()
        
        # 3. In-Memory BM25 Sparse Corpus
        self.bm25_chunks: List[Dict[str, Any]] = []
        self.bm25_model: Optional[BM25Okapi] = None

    def _init_collection(self):
        """Creates the Qdrant collection if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )

    def _convert_to_uuid(self, string_id: str) -> str:
        """Converts string chunk IDs into deterministic UUID strings for Qdrant compliance."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Indexes document chunks generated by Member 1 into Qdrant & BM25."""
        if not chunks:
            print("[VectorStore] Warning: No chunks provided for indexing.")
            return

        print(f"[VectorStore] Generating dense embeddings for {len(chunks)} chunks using {settings.EMBEDDING_MODEL_NAME}...")
        texts = [c["chunk_text"] for c in chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=False, batch_size=32).tolist()

        points = []
        for chunk, emb in zip(chunks, embeddings):
            qdrant_id = self._convert_to_uuid(chunk["chunk_id"])
            
            payload = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "doc_name": chunk["doc_name"],
                "page_number": chunk["page_number"],
                "section": chunk.get("section", ""),
                "chunk_text": chunk["chunk_text"],
                "year": chunk.get("year"),
                "doc_type": chunk.get("doc_type"),
                "metadata": chunk.get("metadata", {})
            }

            points.append(PointStruct(
                id=qdrant_id,
                vector=emb,
                payload=payload
            ))

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[VectorStore] Indexed {len(points)} vectors into Qdrant collection '{self.collection_name}'.")

        self.bm25_chunks = chunks
        tokenized_corpus = [re.findall(r'\w+', c["chunk_text"].lower()) for c in chunks]
        self.bm25_model = BM25Okapi(tokenized_corpus)
        print(f"[VectorStore] BM25 lexical index successfully built.")

    def _build_qdrant_filter(self, metadata_filter: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Maps key-value user metadata filters into Qdrant Filter objects."""
        if not metadata_filter:
            return None

        conditions = []
        for key, value in metadata_filter.items():
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
        return Filter(must=conditions)

    def search(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search (Dense Vector + BM25 Lexical) using Reciprocal Rank Fusion (RRF).
        Enforces cosine similarity thresholding and payload metadata filtering.
        """
        # 1. Dense Vector Search via Qdrant query_points
        query_vector = self.encoder.encode(query).tolist()
        q_filter = self._build_qdrant_filter(metadata_filter)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k * 5,
            query_filter=q_filter
        )
        vector_results = response.points

        # Apply similarity threshold check to dense results
        valid_dense_hits = [res for res in vector_results if res.score >= similarity_threshold]

        # Early exit: If no dense vector hit satisfies similarity_threshold, query is out-of-domain
        if not valid_dense_hits:
            return []

        valid_chunk_ids = {res.payload["chunk_id"] for res in valid_dense_hits}
        dense_score_map = {res.payload["chunk_id"]: float(res.score) for res in valid_dense_hits}

        if not use_hybrid or not self.bm25_model:
            final_chunks = []
            for hit in valid_dense_hits[:top_k]:
                item = hit.payload.copy()
                item["similarity_score"] = round(float(hit.score), 4)
                final_chunks.append(item)
            return final_chunks

        # 2. Sparse Lexical Search via BM25
        tokenized_query = re.findall(r'\w+', query.lower())
        bm25_scores = self.bm25_model.get_scores(tokenized_query) if tokenized_query else []
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_k = 60.0

        for rank, hit in enumerate(valid_dense_hits):
            c_id = hit.payload["chunk_id"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + settings.HYBRID_DENSE_WEIGHT * (1.0 / (rrf_k + (rank + 1)))
            payload_with_score = hit.payload.copy()
            payload_with_score["similarity_score"] = round(float(hit.score), 4)
            chunk_map[c_id] = payload_with_score

        if len(bm25_scores) > 0:
            bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
            bm25_rank = 1
            for idx in bm25_ranked_indices:
                score = bm25_scores[idx]
                if score <= 0:
                    break
                
                chunk = self.bm25_chunks[idx]
                c_id = chunk["chunk_id"]

                # Enforce thresholding on BM25 candidates: only fuse chunks that passed dense threshold
                if c_id not in valid_chunk_ids:
                    continue
                
                if metadata_filter:
                    match = all(
                        chunk.get(k) == v or chunk.get("metadata", {}).get(k) == v 
                        for k, v in metadata_filter.items()
                    )
                    if not match:
                        continue

                rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + settings.HYBRID_SPARSE_WEIGHT * (1.0 / (rrf_k + bm25_rank))
                if c_id not in chunk_map:
                    payload_with_score = chunk.copy()
                    payload_with_score["similarity_score"] = round(dense_score_map.get(c_id, 0.0), 4)
                    chunk_map[c_id] = payload_with_score
                
                bm25_rank += 1
                if bm25_rank > top_k * 3:
                    break

        # 4. Sort Fusion Candidates & Return Top-K
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        results = []
        for cid in sorted_ids[:top_k]:
            res_item = chunk_map[cid]
            res_item["hybrid_rrf_score"] = round(rrf_scores[cid], 6)
            results.append(res_item)

        return results
```

## File: .gitignore
```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so

# Environments
venv/
.venv/
env/
ENV/
env.bak/
venv.bak/

# Environment Variables & Secrets
.env
.env.local
*.env

# Hugging Face & Sentence Transformers Local Cache
.cache/
huggingface/
transformers_cache/
torch/

# Vector Store Storage (Qdrant on-disk artifacts)
qdrant_data/
*.qdrant
*.db
*.sqlite

# Ingestion Source Documents (Do not commit binary client docs in prod)
documents/*.pdf
!documents/README.md

# OS-Specific
.DS_Store
Thumbs.db
desktop.ini

# IDE & Editor files
.vscode/
.idea/
*.swp
*.swo

# Ignore all files and contents inside documents/
documents/*

# Keep the directory tracked in git (optional, via .gitkeep)
!documents/.gitkeep
```

## File: generate_test_docs.py
```python
import pymupdf
from pathlib import Path

docs_dir = Path("./documents")
docs_dir.mkdir(parents=True, exist_ok=True)

# 10 Multi-Page Document Definitions (Minimum 5 pages each)
DOCS_DATA = [
    {
        "filename": "ev_battery_standard_r100_rev2021.pdf",
        "title": "ECE R100 Rev 2021: Specific Requirements for Electric Powertrain Safety",
        "doc_id": "STD-R100-2021",
        "year": 2021,
        "pages": [
            [
                "ECE Regulation No. 100 Revision 2021: Uniform Provisions Concerning Vehicle Safety.",
                "1. Scope and Field of Application. This standard applies to the electric power train of road vehicles of categories M and N.",
                "Section 1.1: High Voltage Safety Protocols. Electrical components must maintain a nominal working voltage exceeding 60 V DC and up to 1500 V DC.",
                "All live high voltage parts must possess IPXXB touch protection under normal operating conditions. Isolation resistance must be at least 100 ohms per volt for DC circuits."
            ],
            [
                "2. Battery Crash Protection and Mechanical Shock Standards (2021 Release).",
                "Paragraph 2.1: Mechanical impact deceleration testing requires a minimum pulse duration of 65 milliseconds with peak acceleration reaching 28g.",
                "Paragraph 2.2: Structural integrity verification dictates that no electrolyte leakage shall occur within 30 minutes following dynamic deceleration.",
                "Deformation of the outer enclosure must not penetrate internal cell casing boundaries under 100 kN static compressive load."
            ],
            [
                "3. Thermal Runaway Propagation Standards (2021 Baseline Thresholds).",
                "Paragraph 3.1: Initiation criteria. A single cell thermal runaway is triggered via a 200W ceramic heating cartridge.",
                "Paragraph 3.2: Mitigation time window. The Battery Management System (BMS) must provide an audible alarm at least 5 minutes prior to hazardous conditions.",
                "The 2021 standard does not mandate complete suppression of fire outside the pack enclosure after 10 minutes of propagation."
            ],
            [
                "4. Chemical Composition and Heavy Metal Limitations (2021 Guidelines).",
                "Section 4.1: Cobalt utilization threshold. Cathode formulations must not exceed 20% cobalt weight ratio in automotive commercial traction cells.",
                "Section 4.2: Recycled materials quota. In 2021, recycled lithium recovery targets remain voluntary with an advisory target of 25% recovery efficiency.",
                "Nickel-Manganese-Cobalt (NMC 622) is established as the certified reference chemistry for compliance bench tests."
            ],
            [
                "5. Verification Annex and Sign-off Criteria (2021 Cycle).",
                "Annex A: State of Health (SoH) degradation thresholds allow a 25% loss over 8 years or 160,000 kilometers.",
                "Annex B: Type approval certification requires testing across 5 production-line battery packs under ambient conditions of 20°C ± 5°C.",
                "Authority sign-off issued under Standard ECE-R100-2021. All revisions supersede 2018 guidelines."
            ]
        ]
    },
    {
        "filename": "ev_battery_standard_r100_rev2023.pdf",
        "title": "ECE R100 Rev 2023: Advanced Thermal Safety and Fast Charging Revisions",
        "doc_id": "STD-R100-2023",
        "year": 2023,
        "pages": [
            [
                "ECE Regulation No. 100 Revision 2023: Addendum on Ultra-Fast Charging and High Current Safety.",
                "1. Scope and Revisions. Supersedes the 2021 baseline with stringent protocols for high-rate charging infrastructures above 250 kW.",
                "Section 1.1: DC Isolation Monitoring. Insulation resistance requirement is raised from 100 ohms/volt to 500 ohms/volt under humid operating conditions.",
                "BMS ground fault detection circuits must trigger high-voltage disconnect contactors within 100 milliseconds of threshold breach."
            ],
            [
                "2. Dynamic Mechanical Crush and Penetration Updates (2023 Standards).",
                "Paragraph 2.1: Nail penetration testing is reinstated for non-LFP chemistries using a 5mm tungsten carbide pin moving at 20 mm/sec.",
                "Paragraph 2.2: Enclosure ingress rating is formally upgraded to IP6K9K to ensure resistance against high-pressure steam jet cleaning.",
                "Electrolyte leakage after static 100 kN crush test is strictly prohibited; zero milliliter discharge is the absolute pass criterion."
            ],
            [
                "3. Extended Thermal Runaway Protection (2023 Enforcement).",
                "Paragraph 3.1: The driver early warning window is extended from 5 minutes to 15 minutes prior to smoke or gas ingress into the passenger cabin.",
                "Paragraph 3.2: Venting gas flammability. Pack enclosures must incorporate directional exhaust burst discs designed to vent away from passenger zones.",
                "Internal cell-to-cell thermal barrier aerogel blankets must sustain 800°C for a minimum duration of 20 minutes."
            ],
            [
                "4. Heavy Metal Ceilings and Material Tracking (2023 Revision).",
                "Section 4.1: Cobalt reduction limit. Cobalt weight ratio in NMC cathodes is strictly capped at 10% (transitioning toward NMC 811 formulations).",
                "Section 4.2: Closed-loop recycling quotas. Manufacturers must demonstrate that at least 50% of active lithium and 70% of cobalt are recovered during end-of-life.",
                "Battery passport digital traceability is introduced as a voluntary audit provision."
            ],
            [
                "5. Warranty and Capacity Retention Standards (2023 Requirements).",
                "Annex A: Capacity retention mandate is tightened: maximum allowable degradation is 20% over 8 years or 160,000 kilometers.",
                "Annex B: High-rate fast charging test mandates 500 consecutive cycles at 3C rate with less than 8% non-linear degradation.",
                "Compliance sign-off registered under ECE-R100-2023, superseding the 2021 protocol."
            ]
        ]
    },
    {
        "filename": "ev_battery_standard_r100_rev2026.pdf",
        "title": "ECE R100 Rev 2026: Solid-State Integration and Zero-Propagation Mandate",
        "doc_id": "STD-R100-2026",
        "year": 2026,
        "pages": [
            [
                "ECE Regulation No. 100 Revision 2026: Solid-State and Advanced Cell Architecture Framework.",
                "1. Scope: Formal guidelines encompassing both liquid electrolyte lithium-ion and solid-state electrolyte battery architectures.",
                "Section 1.1: Operating Voltage Ceiling. Standardized protocols updated for 1200 V DC high-voltage architectures.",
                "Isolation monitoring systems must execute self-diagnostic tests continuously every 10 milliseconds, maintaining 1000 ohms per volt."
            ],
            [
                "2. Zero-Propagation Thermal Safety Mandate (2026 Definitive Rule).",
                "Paragraph 2.1: The 15-minute warning window from 2023 is replaced by a complete 'Zero-Propagation' standard.",
                "Under thermal runaway of cell #1, no propagation to neighboring cells is permissible under any circumstances.",
                "Passenger compartment warning time is increased to 30 minutes, during which external pack surface temperatures must not exceed 60°C."
            ],
            [
                "3. Advanced Mechanical Integrity and Submersion Standards (2026 Revision).",
                "Paragraph 3.1: Saltwater submersion test. The entire energized pack is immersed in 3.5% NaCl solution for 2 hours at depth of 1 meter.",
                "No explosion, fire, or arc discharge is permitted during submersion or the subsequent 24-hour observation window.",
                "Mechanical crush criteria require pack structural withstand capability of 150 kN omnidirectional force."
            ],
            [
                "4. Cathode Sustainability and Battery Passport (2026 Mandatory Standards).",
                "Section 4.1: Cobalt ceiling. Cobalt content must not exceed 5% by total active cathode mass. Cobalt-free (LFP, LMFP, Na-ion) formats incentivized.",
                "Section 4.2: Mandatory Battery Passport. Every commercial pack sold in 2026 must encode an immutable cryptographic passport detailing carbon footprint.",
                "Recycling mandate requires 80% lithium recovery and 95% nickel/cobalt recovery efficiency verified by third-party audit."
            ],
            [
                "5. Lifecycle Guarantees and Certification Protocols (2026 Release).",
                "Annex A: Capacity retention guarantees: maximum allowable degradation is limited to 15% over 10 years or 200,000 kilometers.",
                "Annex B: Solid-state dendrite penetration testing requires 1200 high-voltage pulsed charge cycles without internal micro-shorts.",
                "Issued under Authority Registry ECE-R100-2026. Prior iterations (2021, 2023) are designated legacy status."
            ]
        ]
    },
    {
        "filename": "iso_6469_part1_safety_2022.pdf",
        "title": "ISO 6469-1:2022 Electrically Propelled Road Vehicles - Safety Specifications",
        "doc_id": "ISO-6469-1-2022",
        "year": 2022,
        "pages": [
            [
                "ISO 6469-1:2022 International Organization for Standardization - Rechargeable Energy Storage Systems (RESS).",
                "Clause 1: Scope. Specifies electrical safety requirements for secondary battery packs mounted in wheeled electric vehicles.",
                "Clause 1.2: Overcurrent Interruption. Protection devices must sever 10 kA short-circuit current within 2 milliseconds without explosive rupture.",
                "Current sensor accuracy across dynamic discharge phases must remain within ±0.5% full-scale tolerance."
            ],
            [
                "Clause 2: Isolation Resistance Testing in Wet Conditions (2022 Methodology).",
                "Subclause 2.1: Preconditioning. The battery pack is subjected to 95% relative humidity at 40°C for 48 consecutive hours.",
                "Subclause 2.2: Measurement. Minimum insulation resistance between live components and chassis ground must exceed 500 ohms/volt.",
                "Equipotential bonding resistance between exposed conductive parts and structural vehicle ground must not exceed 0.1 ohm."
            ],
            [
                "Clause 3: Temperature Range and Thermal Runaway Containment (2022 Benchmarks).",
                "Subclause 3.1: Operational ambient temperature envelope is defined from -30°C to +55°C.",
                "Thermal management cooling fluid circuits must withstand proof pressure of 2.0 bar without micro-fissure leakage.",
                "Ethylene glycol based liquid coolant must maintain dielectric breakdown voltage greater than 25 kV."
            ],
            [
                "Clause 4: Gas Venting and Toxic Effluent Management (2022 Standards).",
                "Subclause 4.1: Emission control. In the event of single-cell venting, emissions of hydrogen fluoride (HF) must remain below 3 ppm inside the passenger cell.",
                "Carbon monoxide (CO) concentrations inside the cabin space must not exceed 50 ppm measured over an 8-hour time-weighted average.",
                "Passive pressure relief burst foils must trigger between 0.2 bar and 0.5 bar internal differential pressure."
            ],
            [
                "Clause 5: Final Inspection, Traceability, and Safety Documentation (2022 Norms).",
                "Annex C: Documentation requirements mandate delivery of Material Safety Data Sheets (MSDS) with UN 38.3 test summaries attached.",
                "Factory end-of-line testing requires high-potential (HiPot) dielectric withstand testing at 2 x Nominal Voltage + 1000 V for 60 seconds.",
                "Standard approved under ISO TC22/SC37 Committee, Revision 2022."
            ]
        ]
    },
    {
        "filename": "iso_6469_part1_safety_2025.pdf",
        "title": "ISO 6469-1:2025 Electrically Propelled Road Vehicles - Next-Gen Safety Protocols",
        "doc_id": "ISO-6469-1-2025",
        "year": 2025,
        "pages": [
            [
                "ISO 6469-1:2025 Comprehensive Revisions to Rechargeable Energy Storage Systems Safety Specifications.",
                "Clause 1: Scope Expansion. Formally incorporates bi-directional Vehicle-to-Grid (V2G) continuous cycling stress profiles.",
                "Clause 1.2: Short Circuit Protection. Pyrotechnic circuit disconnect fuses (Pyro-fuses) are made mandatory for packs exceeding 400V nominal.",
                "The disconnect actuation window under hard dead-short fault is reduced from 2.0 ms to 0.8 milliseconds."
            ],
            [
                "Clause 2: High-Voltage Isolation and Moisture Barrier Testing (2025 Updates).",
                "Subclause 2.1: Preconditioning now requires cyclic thermal shock (-40°C to +85°C) combined with salt fog mist exposure for 96 hours.",
                "Insulation resistance must remain above 1000 ohms/volt throughout dynamic charging cycles.",
                "Chassis ground bonding resistance is lowered to a maximum allowable 0.05 ohm."
            ],
            [
                "Clause 3: Active Thermal Safety and Immersion Cooling (2025 Provisions).",
                "Subclause 3.1: Immersion cooling standards. Direct contact dielectric fluids (fluorinated hydrocarbons or synthetic hydrocarbons) are codified.",
                "Dielectric fluid breakdown rating must exceed 40 kV. Flash point of active immersion coolant must be greater than 220°C.",
                "Cooling system pump redundancies require dual independent electric drive motors."
            ],
            [
                "Clause 4: Toxic Gas and Flammable Byproduct Exhaust Limits (2025 Limits).",
                "Subclause 4.1: Cabin effluent limits tightened. Hydrogen fluoride (HF) concentration must not exceed 1 ppm in passenger cabin.",
                "Cabin carbon monoxide (CO) exposure must remain below 25 ppm during any venting failure scenario.",
                "Mandatory active exhaust filtration or catalytic neutralization traps inside pack vent ducts."
            ],
            [
                "Clause 5: Quality Assurance and In-Field Telematics Reporting (2025 Rules).",
                "Annex C: Continuous cloud telematics reporting of cell impedance divergence is mandatory for fleet commercial operation.",
                "HiPot withstand voltage raised to 2.5 x Nominal Voltage + 1200 V for 60 seconds at factory QA sign-off.",
                "Standard certified under ISO TC22/SC37 Committee, Revision 2025, superseding ISO 6469-1:2022."
            ]
        ]
    },
    {
        "filename": "un383_transport_safety_rev2020.pdf",
        "title": "UN Manual of Tests and Criteria Part 38.3 Rev 2020: Transport of Lithium Batteries",
        "doc_id": "UN-383-REV2020",
        "year": 2020,
        "pages": [
            [
                "United Nations Recommendations on the Transport of Dangerous Goods: UN 38.3 Edition 2020.",
                "Section 38.3.1: Scope. Covers transport safety certification for lithium metal and lithium ion cells and batteries.",
                "All lithium ion cells must satisfy tests T.1 through T.8 prior to commercial freight shipment.",
                "Battery assemblies exceeding 12 kg gross weight are categorized as large battery systems."
            ],
            [
                "Section 38.3.2: Altitude Simulation (Test T.1) and Thermal Shock Cycling (Test T.2).",
                "T.1 Altitude: Cells are stored at an absolute pressure of 11.6 kPa (simulating 15,000 meters altitude) for a minimum of 6 hours.",
                "Criterion: No mass loss exceeding 0.1%, no leakage, no venting, and open circuit voltage retention not less than 90%.",
                "T.2 Thermal Test: 10 cycles between -40°C and +72°C with thermal transition time under 30 minutes."
            ],
            [
                "Section 38.3.3: Vibration (Test T.3) and Mechanical Shock (Test T.4) (2020 Protocols).",
                "T.3 Vibration: Sinusoidal sweep from 7 Hz to 200 Hz back to 7 Hz traversed in 15 minutes, repeated for 12 sweeps in each of 3 axes.",
                "T.4 Shock: Half-sine shock of 150g peak acceleration with pulse duration of 6 milliseconds for small cells, 50g/11ms for large packs.",
                "Pass requirement: Zero fire, zero disassembly, and voltage stability across all terminals."
            ],
            [
                "Section 38.3.4: External Short Circuit (Test T.5) and Overcharge (Test T.7).",
                "T.5 Short Circuit: External resistance less than 0.1 ohm applied at 57°C ± 4°C until case temperature returns to 57°C.",
                "Case temperature must not exceed 170°C during test T.5 for compliance under 2020 rules.",
                "T.7 Overcharge: Minimum charge current of 2x manufacturer rated continuous charge current for 24 hours."
            ],
            [
                "Section 38.3.5: State of Charge (SoC) Shipping Ceilings (2020 Regulations).",
                "Air freight restriction: Dedicated air cargo transport of standalone lithium-ion batteries requires SoC capped at 30% nominal.",
                "Maritime vessel transport allows shipments at up to 50% State of Charge under Class 9 Dangerous Goods cargo declaration.",
                "UN 38.3 Rev 2020 Summary Table issued by certified UN testing laboratory."
            ]
        ]
    },
    {
        "filename": "un383_transport_safety_rev2024.pdf",
        "title": "UN Manual of Tests and Criteria Part 38.3 Rev 2024: Maritime and Air Cargo Limits",
        "doc_id": "UN-383-REV2024",
        "year": 2024,
        "pages": [
            [
                "United Nations Manual of Tests and Criteria: Seventh Revised Edition, Amendment 2 (UN 38.3 Rev 2024).",
                "Section 38.3.1: Scope Revision. Includes updated mandates for Sodium-Ion battery chemistries alongside lithium technologies.",
                "All test regimens (T.1 to T.8) apply to sodium-ion cells exceeding 1.2V open circuit potential.",
                "Establishes strict definitions for damaged, defective, or end-of-life battery pack shipments."
            ],
            [
                "Section 38.3.2: Upgraded Thermal Test (T.2) and Overcharge Safeguards (T.7).",
                "T.2 Revision: Dwell time at extreme temperature plateaus (-40°C and +72°C) increased to 8 hours per plateau across 12 full cycles.",
                "Voltage retention baseline is raised from 90% to 92% residual open circuit potential.",
                "T.7 Overcharge testing requires secondary BMS hardware fault injection during initiation."
            ],
            [
                "Section 38.3.3: External Short Circuit Extreme Benchmark (Test T.5 Updates 2024).",
                "External short circuit resistance tightened to less than 0.05 ohm (50 milliohms).",
                "The maximum allowable cell case surface temperature is reduced from 170°C to 150°C.",
                "Test must be monitored for 6 hours post-circuit disconnection to verify delayed thermal reactions."
            ],
            [
                "Section 38.3.4: Maritime Bulk Container Transport Constraints (2024 Revision).",
                "Maritime shipping rules under the International Maritime Dangerous Goods (IMDG) Code are updated.",
                "The historical 50% maritime State of Charge allowance is formally repealed: maximum allowable maritime SoC is capped at 30%.",
                "Mandatory installation of temperature sensors logging every 15 minutes inside refrigerated shipping containers (reefers)."
            ],
            [
                "Section 38.3.5: Damaged / Recycled Cell Freight Safety Protocols (2024 Rules).",
                "Damaged or defective batteries (SP 376) must be transported in non-combustible vermiculite packaging rated for pyro-containment.",
                "Packaging must withstand an internal deflagration temperature of 1000°C for 30 minutes without external case breach.",
                "Standard validated under UN Dangerous Goods Transport Subcommittee Resolution 2024."
            ]
        ]
    },
    {
        "filename": "us_dot_nhtsa_battery_fmvss_2022.pdf",
        "title": "NHTSA FMVSS 305 Rev 2022: Electric Vehicle Crash and Electrical Spill Rules",
        "doc_id": "NHTSA-FMVSS305-2022",
        "year": 2022,
        "pages": [
            [
                "US Department of Transportation: NHTSA Federal Motor Vehicle Safety Standard (FMVSS) No. 305 (2022 Enforcement).",
                "1. Purpose: Mitigate fatalities and injuries from electrical shock and electrolyte burn hazards during and after motor vehicle collisions.",
                "Application: All passenger cars, multipurpose vehicles, and light trucks with electrical components operating above 60 V DC.",
                "Standard barrier impact crashes executed at 35 mph (56 km/h) into fixed frontal concrete barriers."
            ],
            [
                "2. Electrolyte Spillage Limitations (FMVSS 305-2022 Thresholds).",
                "Clause 2.1: Post-crash electrolyte spillage outside the vehicle must not exceed 5.0 liters within the first 30 minutes post-impact.",
                "Inside the passenger compartment, zero liquid electrolyte leakage is permitted under any collision orientation.",
                "Spillage collection trays must be installed beneath battery structures during dynamic testing."
            ],
            [
                "3. Battery Enclosure Retention and Mechanical Attachment (2022 Criteria).",
                "Clause 3.1: Retention. The battery assembly must remain anchored to the vehicle chassis rails.",
                "No mounting bracket failure or complete structural separation from floor pan cross-members is allowed under side-pole impact (20 mph).",
                "Intrusion of chassis rails into cell modules must not exceed 20 mm."
            ],
            [
                "4. Electrical Isolation Verification Post-Impact (2022 Standards).",
                "Clause 4.1: Post-collision electrical isolation of high-voltage DC busbars must be greater than 500 ohms per volt.",
                "Alternatively, if high voltage isolation is lost, voltage on all exposed cables must discharge below 60 V DC within 5.0 seconds.",
                "Capacitive energy storage retention on disconnected lines must drop below 0.2 Joules."
            ],
            [
                "5. Administrative Certification and Compliance Filings (2022 Code).",
                "Section 5: Manufacturers must submit dynamic test crash telemetry to NHTSA Office of Vehicle Safety Compliance within 60 days.",
                "All vehicles must feature an easily accessible manual First Responder Loop physically severing high-voltage contactor circuits.",
                "FMVSS 305 Rev 2022 certified by US Department of Transportation."
            ]
        ]
    },
    {
        "filename": "us_dot_nhtsa_battery_fmvss_2025.pdf",
        "title": "NHTSA FMVSS 305 Rev 2025: Severe Angle Crash and Fire Suppression Rules",
        "doc_id": "NHTSA-FMVSS305-2025",
        "year": 2025,
        "pages": [
            [
                "US Department of Transportation: NHTSA FMVSS No. 305 Revisions and High-Energy Safety Enhancements (2025).",
                "1. Scope Expansion: Incorporates commercial medium and heavy-duty vehicles (Class 4 through Class 8 electric trucks).",
                "Frontal impact speed test requirement raised from 35 mph to 40 mph (64 km/h) against 50% overlap deformable barriers.",
                "Side impact pole velocity increased to 25 mph with mandatory oblique angle (75-degree) impact verification."
            ],
            [
                "2. Zero-Tolerance Electrolyte Discharge Rules (2025 Overhaul).",
                "Clause 2.1: The 2022 allowance of 5.0 liters external electrolyte leakage is eliminated.",
                "Under the 2025 standard, maximum allowable electrolyte spillage is exactly 0.0 liters (zero discharge permitted).",
                "Any liquid detection outside battery pack seals constitutes an automatic compliance failure."
            ],
            [
                "3. Mandatory High-Voltage Discharge Acceleration (2025 Speed Requirements).",
                "Clause 3.1: Active discharge mechanism. High-voltage bus residual charge must discharge below 60 V DC in less than 2.0 seconds.",
                "This updates the prior 5.0-second allowance from the 2022 regulation to protect trapped occupants and emergency rescue workers.",
                "Energy stored in DC filter capacitors must dissipate below 0.1 Joules within the 2.0-second window."
            ],
            [
                "4. Active Pyrotechnic Fire Suppression Systems (2025 Innovation Mandate).",
                "Clause 4.1: Traction batteries exceeding 80 kWh must integrate internal active fire extinguishing or aerosol suppression systems.",
                "Suppression systems must flood battery cell compartments with potassium aerosol or clean gas within 500 milliseconds of thermal runaway detection.",
                "Containment system must prevent reignition for at least 60 minutes post-impact."
            ],
            [
                "5. Safety Filing and Automated Telematics Registry (2025 Requirements).",
                "Section 5: Real-time automatic collision notification (eCall) must transmit battery high-voltage disconnection confirmation to 911 dispatchers.",
                "Annual OEM recertification testing required for every battery pack revision altering cell chemistry or structural casing.",
                "Promulgated by NHTSA Executive Directive, Standard FMVSS 305-2025."
            ]
        ]
    },
    {
        "filename": "eu_battery_directive_sustainability_2024.pdf",
        "title": "EU Battery Regulation 2024/1542: Carbon Footprint and Due Diligence",
        "doc_id": "EU-BAT-2024",
        "year": 2024,
        "pages": [
            [
                "Regulation (EU) 2024/1542 of the European Parliament and of the Council concerning Batteries and Waste Batteries.",
                "Chapter I: General Provisions and Scope. Applies to all categories of batteries: portable, starting/lighting, and electric vehicle batteries.",
                "Article 1: Sets targets for carbon footprint declarations, recycled content thresholds, and performance durability criteria across member states.",
                "By August 2024, all EV batteries entering the European market must possess a verified Carbon Footprint Declaration."
            ],
            [
                "Chapter II: Carbon Footprint Calculation and Maximum Lifecycle Limits.",
                "Article 7: Calculation methodology encompasses upstream raw material acquisition, transport, cell manufacturing, and pack assembly.",
                "Carbon intensity is reported as kilograms of CO2 equivalent per kilowatt-hour of total battery storage capacity (kg CO2e/kWh).",
                "Starting December 2024, batteries exceeding the third-party verified maximum threshold of 110 kg CO2e/kWh cannot be registered."
            ],
            [
                "Chapter III: Mandatory Minimum Recycled Material Contents (2024 Milestones).",
                "Article 8: Minimum shares of recovered material present in active battery materials are legally binding.",
                "Mandatory targets: 16% recovered Cobalt, 85% recovered Lead, 6% recovered Lithium, and 6% recovered Nickel.",
                "Manufacturers must maintain chain-of-custody documentation certified via European Union notified inspection bodies."
            ],
            [
                "Chapter IV: Removability, Replaceability, and Battery Health Transparency.",
                "Article 11: Electric vehicle batteries must allow diagnosis and module-level repair by certified independent operators.",
                "Battery Management System telemetry must provide open API access for the user to query State of Certified Energy (SoCE).",
                "Cell degradation telemetry must be preserved in permanent flash memory for vehicle operational lifespan."
            ],
            [
                "Chapter V: The Digital European Battery Passport (Implementation Framework).",
                "Article 65: Mandatory implementation date set for February 2027, with pilot data registries commencing registration in 2024.",
                "Passport must store cell chemistry details, manufacturing location, recycled material percentages, and full carbon footprint audit.",
                "Enforced across all 27 EU Member States under Official Journal of the European Union Directive 2024/1542."
            ]
        ]
    }
]

def generate_pdf(doc_spec: dict):
    doc = pymupdf.open()
    
    # 5 Pages minimum
    for page_idx, page_paragraphs in enumerate(doc_spec["pages"]):
        page = doc.new_page(width=595, height=842) # A4 format
        
        # Header block
        header_text = f"DOCUMENT: {doc_spec['filename']} | REGULATION CODE: {doc_spec['doc_id']} | YEAR: {doc_spec['year']}"
        page.insert_textbox(pymupdf.Rect(50, 30, 545, 50), header_text, fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))
        
        # Title block on Page 1
        y_cursor = 60
        if page_idx == 0:
            page.insert_textbox(pymupdf.Rect(50, y_cursor, 545, y_cursor + 45), doc_spec["title"], fontsize=14, fontname="hebo")
            y_cursor += 55
            
        # Body Paragraphs
        for p in page_paragraphs:
            rect = pymupdf.Rect(50, y_cursor, 545, y_cursor + 90)
            page.insert_textbox(rect, p, fontsize=10.5, fontname="helv", lineheight=1.3)
            y_cursor += 95
            
        # Footer block (Pagination)
        footer_text = f"Page {page_idx + 1} of {len(doc_spec['pages'])} | Standard Controlled Copy - Compliance Registry"
        page.insert_textbox(pymupdf.Rect(50, 800, 545, 820), footer_text, fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))

    out_path = docs_dir / doc_spec["filename"]
    doc.save(out_path)
    doc.close()
    print(f"Generated ({len(doc_spec['pages'])} pages): {out_path.name}")

def main():
    print("=" * 70)
    print("GENERATING 10 DETAILED MULTI-PAGE REGULATORY PDF DOCUMENTS (≥ 5 PAGES EACH)")
    print("=" * 70)
    
    # Clean previous test files in documents/
    for existing_file in docs_dir.glob("*.pdf"):
        existing_file.unlink()
        
    for doc_spec in DOCS_DATA:
        generate_pdf(doc_spec)
        
    print("=" * 70)
    print("All 10 documents generated successfully in ./documents/")

if __name__ == "__main__":
    main()
```

## File: ingest_documents.py
```python
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
```

## File: test_rag_engine.py
```python
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
```

## File: test_vector_store.py
```python
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
```

## File: requirements.txt
```
# PDF Extraction & Document Processing
pymupdf>=1.24.0

# Sentence Segmentation & Semantic Embedding Generation
sentence-transformers>=3.0.0
numpy>=1.24.0
scikit-learn>=1.4.0

# Member 2: Vector DB & Lexical Search
qdrant-client>=1.8.0
rank-bm25>=0.2.2

# Member 3 & 4: LLM Service & Backend API
groq>=0.5.0
fastapi>=0.110.0
uvicorn>=0.28.0

# Configuration, Validation & Environment
pydantic>=2.6.0
pydantic-settings>=2.2.0
python-dotenv>=1.0.1
```

## File: app/config.py
```python
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
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = ""
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
```
