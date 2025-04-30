import streamlit as st
import json
import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
import requests
from datetime import datetime

# =============================
# Load NLP Classification Model (BERT)
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
# Load ICH Guidelines from JSON
# =============================
def load_guideline_data():
    path = "ich_guidelines_full_combined.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def search_guidelines(query):
    guidelines = load_guideline_data()
    query_lower = query.lower()
    results = []

    for item in guidelines:
        if (query_lower in item.get("title", "").lower()
            or query_lower in item.get("code", "").lower()
            or query_lower in item.get("purpose", "").lower()
            or query_lower in item.get("for_beginners", "").lower()):
            
            subs = item.get("sub_guidelines", [])
            sub_text = "\n".join([f"- {s['code']}: {s['title']}" for s in subs]) if subs else "None"

            results.append(f"""
### 📘 {item['code']} – {item['title']}
**Category:** {item['category']}  
**CTD Section:** {item['ctd_section']}  
**Introduced:** {item['introduced']}

🔍 **Purpose:** {item['purpose']}  
🧪 **Used For:** {item['used_for']}  
🧒 **Beginner Tip:** {item['for_beginners']}  
🔗 **Sub-Guidelines:**  
{sub_text}
""")

    return "\n---\n".join(results) if results else "No matching guidelines found."

# =============================
# Fetch Wikipedia Data (Optional)
# =============================
def fetch_online_data(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "No online data found.")
    return "Error fetching online data."

# =============================
# Get Greeting Based on Time
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
# Streamlit App UI
# =============================
st.set_page_config(page_title="ICH Guidelines Chatbot", layout="centered")
st.title("ICH Guidelines Chatbot")
st.markdown("Ask anything about ICH Quality, Safety, Efficacy, and CTD Dossier preparation!")

# Sidebar
with st.sidebar:
    st.header("About This Tool")
    st.write("This chatbot helps you understand and find relevant ICH guidelines easily.")
    st.write("Built by Kailash using BERT + JSON intelligence.")


with st.sidebar:
    st.header("About")
    st.markdown("""
Hi! I’m **Kailash Kothari**, the developer of this chatbot.  
This tool is designed to help you quickly understand and access the right **ICH guidelines** for dossier preparation.

Whether you’re working on Quality (Q), Safety (S), Efficacy (E), or even eCTD formatting — just ask your question, and I’ll guide you.

I built this to simplify regulatory work and make learning ICH a little more interactive. 💼📘
""")

# Clear Chat
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Message History
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input
if prompt := st.chat_input("What would you like to know about ICH guidelines?"):
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
