import requests
from langchain.llms.base import LLM
from typing import List, Optional


class CustomAPILanguageModel(LLM):
    base_url: str
    api_key: str
    model: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Error: {response.status_code} - {response.text}")

    @property
    def _llm_type(self):
        return "custom_api"

    @property
    def identifying_params(self):
        return {"model": self.model}
