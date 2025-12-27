
from typing import List, Dict
from .base_agent import BaseAgent
from prompts import supervisor_system_prompt 

class SupervisorAgent(BaseAgent):
    def decide_agent(self, user_message: str, history: List[Dict]) -> str:
        # Create a small text view of recent history
        history_text_parts = []
        for turn in history[-5:]:
            history_text_parts.append(f"{turn['role']}: {turn['content']}")
        history_text = "\n".join(history_text_parts)

        user_prompt = f"""
User message:
{user_message}

Recent history:
{history_text}
"""

        messages = [
            {"role": "system", "content": supervisor_system_prompt },
            {"role": "user", "content": user_prompt},
        ]

        raw_output = self.chat_completion(messages).strip().lower()
        print(f"[SUPERVISOR] Raw routing decision: {raw_output!r}")

        # Normalize result
        if "bfsi" in raw_output:
            return "bfsi"
        if "general" in raw_output:
            return "general"
        if "unwanted" in raw_output:
            return "unwanted"

        # Fallback
        print("[SUPERVISOR] Could not clearly parse intent. Defaulting to 'general'.")
        return "general"
