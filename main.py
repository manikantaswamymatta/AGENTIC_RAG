from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
from config import get_hf_client, get_hf_config
from rag.vector_store import VectorStore
from agents.supervisor_agent import SupervisorAgent
from agents.bfsi_agent import BFSIAgent
from agents.general_agent import GeneralAgent
from agents.unwanted_agent import UnwantedAgent

class AgenticRAGService:
    """
    Main orchestrator class:
    - Keeps in-memory session chat history
    - Uses supervisor to find intent
    - Routes to correct agent (loan/general/unwanted)
    """
    def __init__(self):
        print("[SERVICE] Initializing AgenticRAGService...")
        self.cfg = get_hf_config()
        self.client = get_hf_client(self.cfg)
        # reading docs and setting up vector store
        self.vector_store = VectorStore()
        self.vector_store.add_pdf("data/sample_loan3.pdf", category="loan")
    
        # Initialize agents
        self.supervisor = SupervisorAgent(self.client, self.cfg)
        self.bfsi_agent = BFSIAgent(self.client, self.cfg, self.vector_store)
        self.general_agent = GeneralAgent(self.client, self.cfg, self.vector_store)
        self.unwanted_agent = UnwantedAgent(self.client, self.cfg)

        # In-memory chat history per session_id
        self.sessions = {}

        print("[SERVICE] AgenticRAGService initialized successfully.")

    def get_history(self, session_id: str) -> List[Dict]:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def handle_user_message(self, session_id: str, user_message: str) -> Dict:
        print(f" New message for session_id={session_id}: {user_message}")
        history = self.get_history(session_id)

        # Append user message to history
        history.append({"role": "user", "content": user_message, "agent": None})
        print(f"[SERVICE] Current history length: {len(history)}")
        intent = self.supervisor.decide_agent(user_message, history)
        intent = intent.strip().lower()
        print(f"[mani] Initial intent from supervisor: {intent}")

        # Route using supervisor
        intent = self.supervisor.decide_agent(user_message, history)
        print(f"[SERVICE] Supervisor decided intent: {intent}")

        # Call appropriate agent
        if intent == "bfsi":
            answer = self.bfsi_agent.handle(user_message, history)
            agent_used = "bfsi_agent"
        elif intent == "general":
            answer = self.general_agent.handle(user_message, history)
            agent_used = "general_agent"
        else:
            answer = self.unwanted_agent.handle(user_message, history)
            agent_used = "unwanted_agent"
        print(f"[SERVICE] Agent {agent_used} provided answer.")


        # Append assistant response to history
        history.append({"role": "assistant", "content": answer, "agent": agent_used})
        print(f"[SERVICE] Appended assistant response from {agent_used}. History length: {len(history)}")
        print(f"history: {history}")
        return {
            "session_id": session_id,
            "intent": intent,
            "agent": agent_used,
            "answer": answer,
        }

# FastAPI setup
# 
app = FastAPI(title="Agentic RAG BFSI POC")

service = AgenticRAGService()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    intent: str
    agent: str
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Simple single endpoint for the POC.

    Request body:
    {
      "session_id": "user123",
      "message": "I want to know about home loan eligibility"
    }
    """
    print("[API] /chat endpoint called.")
    result = service.handle_user_message(
        session_id=request.session_id,
        user_message=request.message,
    )
    print("[API] /chat endpoint completed successfully.")
    return ChatResponse(**result)
