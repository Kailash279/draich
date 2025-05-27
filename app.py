# app.py
import streamlit as st
import json
from RL_GEN_AI import generate_dynamic_response
from databass import (
    load_data as load_guidelines_data,
    search_guidelines,
    update_memory as save_to_memory,
    load_memory as search_memory,
    get_ich_overview,
    learn_from_text
)
from utils import extract_text_from_pdf, summarize_text
from transformers import AutoTokenizer, pipeline

# ========== Initialization ==========
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

MAX_INPUT_LENGTH = tokenizer.model_max_length

# ========== Truncation Helper ==========
def truncate_text(text, max_length=500):
    if not text:
        return ""
    return text if len(text) <= max_length else text[:max_length] + "..."

# ========== Page Setup ==========
st.set_page_config(page_title="ICH Chatbot Assistant", page_icon="🤖")
st.title("ICH Guidelines Assistant 🤖")

# ========== Session State Initialization ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== Load Guidelines Data ==========
@st.cache_resource(show_spinner=False)
def load_data():
    return load_guidelines_data()

data = load_data()

# ========== Helper Functions ==========
def handle_uploaded_file(uploaded_file):
    try:
        text = learn_from_text(uploaded_file)
        return text
    except Exception as e:
        st.error(f"Error extracting text from file: {e}")
        return ""

def respond_to_query(query):
    query_lower = query.lower()

    if "what is ich" in query_lower:
        return get_ich_overview(data)

    mem_response = search_memory(query)
    if mem_response and mem_response.strip().lower() not in ["none", "null", ""]:
        return f"(📁 From Memory)\n\n{mem_response}"

    if any(keyword in query_lower for keyword in ["guideline", "q1", "q2", "s1", "m4"]):
        db_resp = search_guidelines(query, data)
        if db_resp:
            return f"(📘 From ICH Database)\n\n{db_resp}"

    return generate_dynamic_response(query)

# ========== File Upload & Learning ==========
with st.expander("📄 Upload a file to teach the bot (PDF, TXT, Image)"):
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "jpg", "jpeg", "png"])
    if uploaded_file:
        learned_text = handle_uploaded_file(uploaded_file)
        if learned_text.strip():
            summary = generate_dynamic_response(learned_text[:500])
            save_to_memory(learned_text, summary)
            st.success("✅ Bot learned from the uploaded content!")
        else:
            st.warning("Could not extract text from this file.")

# ========== PDF Summary Section ==========
with st.expander("📁 Upload ICH PDF for Summary"):
    summary_file = st.file_uploader("Upload ICH Guideline PDF", type=["pdf"], key="summary_file")
    if summary_file:
        with open("temp.pdf", "wb") as f:
            f.write(summary_file.getbuffer())
        full_text = extract_text_from_pdf("temp.pdf")
        st.success("PDF uploaded and read successfully!")
        short_text = truncate_text(full_text)
        summary = summarizer(short_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        st.subheader("Summary of the Guideline:")
        st.write(summary)

# ========== Chat History ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ========== Chat Input ==========
query = st.chat_input("Ask anything about ICH...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    response = respond_to_query(query)
    save_to_memory(query, response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# ========== Sidebar ==========
with st.sidebar:
    st.header("📘 About This Assistant")
    st.markdown(
        """
        **ICH Guidelines Assistant 🤖**

        Hi, I am **Kailash Kothari**, the developer of this chatbot. 🤝

        This AI Assistant helps you understand and search through ICH Guidelines using:

        - 🧠 Memory-based recall
        - 📘 Structured ICH database
        - 🌐 Wikipedia fallback
        - ♻️ RL learning loop (self-teaching)
        - 📄 Upload files to help the bot learn

        Thanks for trying it out!
        """
    )