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