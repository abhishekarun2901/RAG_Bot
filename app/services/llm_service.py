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