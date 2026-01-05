# benchmark/sessions/vertex_client.py
import os
import time
from .llm_session_generator import SessionLLM

try:
    from google import genai
    from google.genai import types
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False


class VertexLLM(SessionLLM):
    """
    Google Vertex AI / Gemini API client for session generation.
    Requires: pip install google-genai
    Set GOOGLE_API_KEY env var or use application default credentials.
    """
    
    def __init__(self, model: str = "gemini-3-pro-preview"):
        if not VERTEX_AVAILABLE:
            raise ImportError("google-genai not installed. Run: pip install google-genai")
        
        self.model = model
        # Initialize client - uses GOOGLE_API_KEY env var or ADC
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            # Use Application Default Credentials for Vertex AI
            self.client = genai.Client(vertexai=True)
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        system_instruction = (
            "You generate dialogue sessions. Output ONLY alternating user/assistant turns. "
            "Format each line exactly as:\n"
            "user: <message>\n"
            "assistant: <message>\n"
            "No markdown, no extra text, just the dialogue."
        )
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=4000,
                        temperature=0.7,
                    ),
                )
                
                # Extract only text parts, ignoring thought_signature and other non-text parts
                content = ""
                if response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    content += part.text
                
                if not content:
                    print(f"[VertexLLM] Warning: Empty response")
                
                return content
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                print(f"[VertexLLM] Error: {e}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
        
        print(f"[VertexLLM] Failed after {max_retries} retries: {last_error}")
        return ""


# Keep OpenRouter as fallback option
class OpenRouterLLM(SessionLLM):
    def __init__(self, model: str = "google/gemini-3-pro-preview"):
        import requests
        self.requests = requests
        self.api_key = os.environ["OPENROUTER_API_KEY"]
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = self.requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": (
                                "You generate dialogue sessions. Output ONLY alternating user/assistant turns. "
                                "Format each line exactly as:\n"
                                "user: <message>\n"
                                "assistant: <message>\n"
                                "No markdown, no extra text, just the dialogue."
                            )},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 4000,
                        "temperature": 0.7,
                    },
                    timeout=90,
                )
                resp.raise_for_status()
                data = resp.json()
                
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                
                if not content:
                    print(f"[OpenRouterLLM] Warning: Empty response")
                
                return content or ""
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                print(f"[OpenRouterLLM] Error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
        
        print(f"[OpenRouterLLM] Failed after {max_retries} retries: {last_error}")
        return ""
