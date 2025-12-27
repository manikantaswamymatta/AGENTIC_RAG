
from typing import List, Dict
from .base_agent import BaseAgent
from prompts import unwanted_agent_system_prompt

class UnwantedAgent(BaseAgent):
    def handle(self, user_message: str, history: List[Dict]) -> str:
        print("[UNWANTED] Handling out-of-scope query...")
        messages = [
            {"role": "system", "content": unwanted_agent_system_prompt},
            {"role": "user", "content": user_message},
        ]
        answer = self.chat_completion(messages)
        print("[UNWANTED] Generated guardrail response.")
        return answer
