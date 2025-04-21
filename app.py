# app.py

import streamlit as st
import json
import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
import requests
from datetime import datetime

# =============================
# NLP MODEL (BERT)
# =============================
def load_model():
    tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
    return tokenizer, model

tokenizer, model = load_model()

def classify_query(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs).item()
    categories = ["General", "Safety", "Quality", "Efficacy", "Miscellaneous"]
    return categories[predicted_class]

# =============================
# LOCAL JSON DATA SEARCH WITH SUMMARIZATION
# =============================
def load_data():
    json_path = "restructured_guidelines.json"
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
            return data if isinstance(data, list) else []
        except Exception as e:
            print("Error loading JSON:", e)
            return []

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

            if show_full:
                text = content
            else:
                text = summarize(content)

            results.append(f"**{title}**\n{text}")

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
st.markdown("Ask me anything about **ICH Guidelines** (Quality, Safety, Efficacy, etc.)")

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Store chat messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input box
if prompt := st.chat_input("Ask your ICH query here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    greeting = get_greeting()
    category = classify_query(prompt)
    local_results = search_guidelines(prompt)
    online_results = fetch_online_data(prompt)

    response = f"{greeting}\n\n**Category:** `{category}`\n\n**Local Match:**\n{local_results}\n\n**Online Info:**\n{online_results}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write("Hi! I am **Kailash Kothari**, the developer of this chatbot. It helps you find ICH guidelines quickly from local data and Wikipedia.")
