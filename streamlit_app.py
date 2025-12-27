import streamlit as st
import requests
import uuid

# =============================
# CONFIG
# =============================
API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="Agentic RAG – BFSI POC",
    layout="wide",
)

# =============================
# SESSION STATE
# =============================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🧠 Agentic RAG")
st.sidebar.markdown("**Session ID**")
st.sidebar.code(st.session_state.session_id)

st.sidebar.markdown("---")
st.sidebar.markdown("### Available Agents")
st.sidebar.markdown("""
- 🏦 **BFSI Agent**
- 📄 **General Agent**
- 🚫 **Unwanted Agent**
""")

# =============================
# MAIN UI
# =============================
st.title("🤖 Agentic RAG (BFSI)")

st.markdown(
    """
Ask a question related to loans or general BFSI topics.
The **Supervisor Agent** will route your query to the correct agent.
"""
)

# =============================
# CHAT DISPLAY
# =============================
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

        if chat["role"] == "assistant":
            st.caption(
                f"🧭 Intent: **{chat['intent']}** | 🤖 Agent: **{chat['agent']}**"
            )

# =============================
# USER INPUT
# =============================
user_input = st.chat_input("Type your question here...")

if user_input:
    # Show user message immediately
    st.chat_message("user").markdown(user_input)

    payload = {
        "session_id": st.session_state.session_id,
        "message": user_input,
    }

    try:
        with st.spinner("Routing via Supervisor Agent..."):
            response = requests.post(API_URL, json=payload, timeout=60)

        if response.status_code != 200:
            st.error(f"API Error: {response.text}")
        else:
            data = response.json()

            # Save history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
            })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": data["answer"],
                "intent": data["intent"],
                "agent": data["agent"],
            })

            # Render assistant message
            with st.chat_message("assistant"):
                st.markdown(data["answer"])
                st.caption(
                    f"🧭 Intent: **{data['intent']}** | 🤖 Agent: **{data['agent']}**"
                )

    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
