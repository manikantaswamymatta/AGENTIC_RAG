
from typing import List, Dict
from openai import AzureOpenAI
from config import HuggingFaceConfig

class BaseAgent:
    def __init__(self, client: AzureOpenAI, cfg: HuggingFaceConfig):
        self.client = client
        self.cfg = cfg

    def chat_completion(self, messages: List[Dict]) -> str:
        """
        Simple wrapper for Azure OpenAI chat completion.
        """
        print(f"[AGENT] Calling Hugging Face model: {self.cfg.chat_model}")

        try:
            response = self.client.chat.completions.create(
                model=self.cfg.chat_model,
                messages=messages,
                # temperature=temperature,
            )
            text = response.choices[0].message.content
            print("[AGENT] LLM call successful.")
            return text
        except Exception as e:
            print(f"[AGENT][ERROR] LLM call failed: {e}")
            return "Sorry, something went wrong while generating a response."
