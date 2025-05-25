# app.py
import streamlit as st
import json
from RL_GEN_AI import generate_dynamic_response
from databass import (
    load_data as load_guidelines_data,
    search_guidelines,
    update_memory as save_to_memory,
    load_memory as search_memory
)
# ========== Load Guidelines ==========
@st.cache_resource(show_spinner=False)
def load_data():
    return load_guidelines_data()

data = load_data()

# ========== Streamlit UI ==========
st.set_page_config(page_title="ICH Chatbot Assistant", page_icon="🤖")
st.title("ICH Guidelines Assistant 🤖")

# ========== Chat Memory State ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ========== Chat History ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ========== Chat Input ==========
query = st.chat_input("Ask anything about ICH...", key="chatbox")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    # 1. Try from memory
    mem_response = search_memory(query)
    if mem_response:
        response = f"(📁 From Memory)\n\n{mem_response}"

    # 2. Try from local ICH database
    elif any(x in query.lower() for x in ["ich", "guideline", "q1", "s1", "e1", "m1", "what is ich"]):
        db_result = search_guidelines(query, data)
        if db_result:
            response = f"(📘 From ICH Database)\n\n{db_result}"
        else:
            response = generate_dynamic_response(query)

    # 3. Else fallback to dynamic RL-based LLM
    else:
        response = generate_dynamic_response(query)

    # Save to memory and show response
    save_to_memory(query, response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ========== Sidebar ==========
with st.sidebar:
    st.header("📘 About This Assistant ")
    st.markdown("""
**ICH Guidelines Assistant 🤖**

Hii i am Kialash Kothari the developer of this chatbot 
This AI Assistant helps you understand and search through ICH Guidelines using:

- 🧠 **Memory-based recall** (saves previous Q&A for instant future replies)
- 📘 **ICH Local Database** (structured from official guidelines)
- 🌐 **Wikipedia Search** (for general pharma/biotech queries)
- ♻️ **Reinforcement Learning Loop** (auto-learns from new user input)

> Built with ❤️ by Kailash Kothari  
> Version: `ICH-GEN-AI v1.0`

**Try asking:**
- What is ICH?
- Tell me about Q8 guideline
- Show safety guideline
- What is bioequivalence?
""")  # ✅ now properly closed
