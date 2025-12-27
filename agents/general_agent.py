
from typing import List, Dict
from .base_agent import BaseAgent
from prompts import general_agent_system_prompt, rag_user_prompt_template
from rag.vector_store import VectorStore
from .bfsi_agent import format_history  # reuse helper

class GeneralAgent(BaseAgent):
    def __init__(self, client, cfg, vector_store: VectorStore):
        super().__init__(client, cfg)
        self.vector_store = vector_store

    def handle(self, user_message: str, history: List[Dict]) -> str:
        print("[GENERAL] Handling general BFSI query...")
        docs = self.vector_store.similarity_search(user_message, k=3)
        context = "\n\n".join(docs)
        history_text = format_history(history)

        user_prompt = rag_user_prompt_template.format(
            context=context,
            history=history_text,
            question=user_message,
        )

        messages = [
            {"role": "system", "content": general_agent_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = self.chat_completion(messages)
        print("[GENERAL] Generated answer.")
        return answer
