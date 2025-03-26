# main.py
import streamlit as st
from nlp_model import classify_query
from database import search_guidelines
from fetch_data import fetch_online_data
from greetings import get_greeting
import json
import os

# =============================
# STREAMLIT CHATBOT UI
# =============================
st.title("ICH Guidelines Chatbot 🤖")
st.write("Ask me anything about ICH guidelines!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    greeting = get_greeting()
    category = classify_query(prompt)
    local_results = search_guidelines(prompt)
    online_results = fetch_online_data(prompt)
    response = f"{greeting}\n**Category:** {category}\n\n{local_results}\n\n{online_results}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# ABOUT SECTION
with st.sidebar:
    st.header("About")
    st.write(" Hii I AM Kailash KOthari the developar of this Chatbot and The chatbot helps you find ICH guidelines quickly!")

# nlp_model.py
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load NLP Model (BERT)
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    return tokenizer, model

tokenizer, model = load_model()

def classify_query(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs).item()
    categories = ["General", "Safety", "Quality", "Efficacy", "Miscellaneous"]
    return categories[predicted_class]

# database.py
import sqlite3

def connect_db():
    conn = sqlite3.connect("pdf_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS guidelines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        content TEXT
    )""")
    conn.commit()
    return conn, cursor

conn, cursor = connect_db()

def search_guidelines(query):
    cursor.execute("SELECT content FROM guidelines WHERE content LIKE ?", (f"%{query}%",))
    results = cursor.fetchall()
    return "\n".join([result[0] for result in results]) if results else "No matching guidelines found."

# fetch_data.py
import requests

def fetch_online_data(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "No online data found.")
    return "Error fetching online data."

# greetings.py
from datetime import datetime

def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning! ☀️"
    elif hour < 18:
        return "Good Afternoon! ☀️"
    else:
        return "Good Evening! 🌙"
