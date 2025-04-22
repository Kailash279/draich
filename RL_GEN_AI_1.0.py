# app.py

import streamlit as st
import json
import os
import requests
from datetime import datetime
import textract  # for reading various file formats
from RL_GEN_AI_1_0 import generate_response  # integrate RL module

# =============================
# FILE HANDLING & EXTRACTION
# =============================
def extract_text_from_file(uploaded_file):
    file_path = f"/tmp/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        text = textract.process(file_path).decode("utf-8")
        return text.strip()
    except Exception as e:
        return f"Could not extract text: {e}"

# =============================
# NLP CLASSIFIER
# =============================
def classify_query(user_input):
    user_input = user_input.lower()
    if "quality" in user_input:
        return "Quality"
    elif "safety" in user_input:
        return "Safety"
    elif "efficacy" in user_input:
        return "Efficacy"
    elif "development" in user_input or "guideline" in user_input:
        return "General"
    else:
        return "Miscellaneous"

# =============================
# LOAD ENHANCED JSON DATA
# =============================
def load_data():
    json_path = "enhanced_guidelines.json"
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
            return data if isinstance(data, list) else []
        except Exception as e:
            print("Error loading JSON:", e)
            return []

# =============================
# SEARCH GUIDELINES WITH SUMMARIZATION
# =============================
def search_guidelines(query):
    data = load_data()
    results = []

    show_full = any(x in query.lower() for x in ["full", "detail", "long", "explain"])
    query_lower = query.lower()

    def summarize(text, lines=3):
        return "\n".join(text.strip().split("\n")[:lines]) + "\n..."

    for entry in data:
        if query_lower in entry.get("title", "").lower() or \
           any(query_lower in kw.lower() for kw in entry.get("keywords", [])) or \
           query_lower in entry.get("summary", "").lower():

            content = entry.get("content", "")
            summary = entry.get("summary", "No summary available.")
            title = entry.get("title", "Untitled")
            explanation = entry.get("explanation", "")

            text = content if show_full else summarize(content)
            results.append(f"**{title}**\n{explanation}\n{text}")

    return "\n\n".join(results) if results else "No matching guidelines found."

# =============================
# FETCH ONLINE DATA
# =============================
def fetch_online_data(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "No online data found.")
    elif response.status_code == 404:
        return "No Wikipedia article found for this query."
    return "Error fetching online data."

# =============================
# GREETING FUNCTION
# =============================
def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning! ☀️"
    elif hour < 18:
        return "Good Afternoon! ☀️"
    else:
        return "Good Evening! 🌙"

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="ICH Guidelines Chatbot", layout="centered")
st.title("ICH Guidelines Chatbot 🤖")
st.markdown("Ask me anything about **ICH Guidelines** or upload a document for analysis.")

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded_file = st.file_uploader("📂 Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
file_content = ""

if uploaded_file:
    file_content = extract_text_from_file(uploaded_file)
    st.success("📄 File content extracted successfully!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input box
if prompt := st.chat_input("Ask your ICH query or related to uploaded file..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    greeting = get_greeting()
    category = classify_query(prompt)
    local_results = search_guidelines(prompt)
    file_insight = ""

    # If user uploaded something, analyze it
    if uploaded_file and file_content:
        file_insight = f"\n\n📎 Based on uploaded file:\n{generate_response(file_content, prompt)}"

    online_results = fetch_online_data(prompt)

    response = f"{greeting}\n\n**Category:** `{category}`\n\n**Local Match:**\n{local_results}{file_insight}\n\n**Online Info:**\n{online_results}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write("Hi! I am **Kailash Kothari**, the developer of this intelligent chatbot. It can analyze ICH guidelines, uploaded documents, and answer using local + online sources.")
