import os
import requests
from app.config import settings
 

class LLMService:

    def __init__(self, model_name: str = settings.GROQ_MODEL_NAME):

        self.api_key = settings.GROQ_API_KEY 

        if not self.api_key:

            raise ValueError("GROQ_API_KEY environment variable or setting is missing.")

        self.model_name = model_name

        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
 
    def generate_answer(self, prompt: str, system_prompt: str) -> str:

        """Executes a custom HTTP POST request directly to the Groq REST API endpoint."""

        headers = {

            "Authorization": f"Bearer {self.api_key}",

            "Content-Type": "application/json"

        }
 
        payload = {

            "model": self.model_name,

            "messages": [

                {"role": "system", "content": system_prompt},

                {"role": "user", "content": prompt}

            ],

            "temperature": 0.1,

            "max_tokens": 1024

        }
 
        try:

            response = requests.post(

                url=self.api_url,

                headers=headers,

                json=payload,

                timeout=30.0

            )

            response.raise_for_status()

            data = response.json()

            return data["choices"][0]["message"]["content"]
 
        except requests.exceptions.RequestException as e:

            error_detail = ""

            if e.response is not None:

                try:

                    error_detail = f" - Body: {e.response.text}"

                except Exception:

                    pass

            raise RuntimeError(f"Custom Groq API Call Failed: {str(e)}{error_detail}") from e
 