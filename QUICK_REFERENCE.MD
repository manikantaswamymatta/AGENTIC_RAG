User
 │
 │  POST /chat
 ▼
FastAPI
 │
 ▼
AgenticRAGService
 │
 ├── SupervisorAgent (intent detection)
 │        ├─ bfsi
 │        ├─ general
 │        └─ unwanted
 │
 ├── LoanAgent (BFSI + RAG)
 │        └─ VectorStore (Chroma + PDFs)
 │
 ├── GeneralAgent
 │
 └── UnwantedAgent




📁 Project Structure

├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Abstract base agent
│   ├── supervisor_agent.py    # Routes query to correct agent
│   ├── general_agent.py       # Handles general queries
│   ├── loan_agent.py          # Handles loan-related queries
│   └── unwanted_agent.py     # Fallback / rejection agent
│
├── rag/
│   └── vector_store.py        # ChromaDB + embedding logic
│
├── data/
│   ├── sample_loan3.pdf       # Sample BFSI documents
│   └── sample_loan4.pdf
│
├── tests/                     # Unit / integration tests
│
├── streamlit_app.py           # Streamlit UI
├── main.py                    # Entry point (CLI / backend)
├── prompts.py                 # All LLM prompts in one place
├── config.py                  # Environment & app configuration
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
├── QUICK_REFERENCE.md         # Developer quick guide
└── README.md                  # Project documentation
