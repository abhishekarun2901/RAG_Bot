import re
from typing import Tuple


class InputGuardrails:
    # 1. Prompt Injection & Jailbreak Patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above)\s+instructions",
        r"disregard\s+(all\s+)?prior",
        r"system\s+prompt",
        r"reveal\s+(your\s+)?instructions",
        r"you\s+are\s+now\s+a",
        r"jailbreak",
        r"dan\s+mode",
        r"override\s+system",
    ]

    # 2. PII Detection Patterns (Email, Phone, Bearer Tokens/Keys)
    PII_PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "API_KEY": r"(?:sk-|gsk_)[a-zA-Z0-9]{20,}",
    }

    @classmethod
    def validate_and_sanitize(cls, query: str) -> Tuple[bool, str, str]:
        """
        Validates input for prompt injections and sanitizes PII.
        
        Returns:
            Tuple[is_safe (bool), processed_query (str), failure_reason (str)]
        """
        # Clean control characters and trim excessive whitespace
        clean_query = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', query).strip()

        # Check for Prompt Injection
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, clean_query, re.IGNORECASE):
                return False, clean_query, "Disallowed system override attempt detected."

        # Mask PII (Redact sensitive tokens before embedding/retrieval)
        sanitized_query = clean_query
        for pii_type, pattern in cls.PII_PATTERNS.items():
            sanitized_query = re.sub(pattern, f"[{pii_type}_REDACTED]", sanitized_query)

        return True, sanitized_query, ""