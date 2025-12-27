
# 🚀 Quick Start – Agentic RAG (All-in-One)

## 1️⃣ Create Virtual Environment

### Windows
python -m venv venv
venv\Scripts\activate

### Linux / Mac
python3 -m venv venv
source venv/bin/activate


## 2️⃣ Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt


## 3️⃣ Configure .env

Create a `.env` file in the project root and add:

HUGGINGFACEHUB_API_TOKEN=your_huggging_face_access_token(free token available)
HF_CHAT_MODEL=mistralai/Mistral-7B-Instruct-v0.2


## ▶️ Running the Application

uvicorn main:app --reload

### Run Streamlit UI
streamlit run streamlit_app.py

## ✅ Expected Output

- Streamlit app opens in browser
- Upload/query PDF-based loan documents
- Supervisor agent routes query automatically
- Relevant agent responds using RAG

-------------------------------------------------
If Streamlit doesn't open automatically:
http://localhost:8501
-------------------------------------------------
