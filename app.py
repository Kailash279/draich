import streamlit as st
import json
import os
import requests
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
# your other imports...

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="ICH Guidelines Chatbot", layout="centered")

# Now rest of your Streamlit app
st.title("ICH Guidelines Chatbot")
# =============================
# Load BERT Model (Fallback safe)
# =============================
@st.cache_resource
def load_model():
    try:
        tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
        return tokenizer, model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None, None

tokenizer, model = load_model()

def classify_query(text):
    if not tokenizer or not model:
        return "Error loading model"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    categories = ["General", "Safety", "Quality", "Efficacy", "Miscellaneous"]
    return categories[torch.argmax(probs).item()]

# =============================
# Load Guidelines JSON
# =============================
def load_guideline_data():
    path = "ich_guidelines_full_combined.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def search_guidelines(query):
    data = load_guideline_data()
    q = query.lower()
    results = []

    for g in data:
        if (q in g.get("title", "").lower() or
            q in g.get("code", "").lower() or
            q in g.get("purpose", "").lower() or
            q in g.get("for_beginners", "").lower()):
            
            subs = g.get("sub_guidelines", [])
            subs_text = "\n".join([f"- {s['code']}: {s['title']}" for s in subs]) if subs else "None"

            results.append(f"""
### 📘 {g['code']} – {g['title']}
**Category:** {g['category']}  
**CTD Section:** {g['ctd_section']}  
**Introduced:** {g['introduced']}

🔍 **Purpose:** {g['purpose']}  
🧪 **Used For:** {g['used_for']}  
🧒 **Beginner Tip:** {g['for_beginners']}  
🔗 **Sub-Guidelines:**  
{subs_text}
""")

    return "\n---\n".join(results) if results else "⚠️ No matching guidelines found."

# =============================
# Wiki Fallback Search (Optional)
# =============================
def fetch_online_data(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json().get("extract", "No data found.")
    return "🌐 No Wikipedia article found."

# =============================
# Time-Based Greeting
# =============================
def get_greeting():
    h = datetime.now().hour
    return "Good Morning ☀️" if h < 12 else "Good Afternoon ☀️" if h < 18 else "Good Evening 🌙"

# =============================
# Streamlit App UI
# =============================
st.set_page_config(page_title="ICH Guidelines Chatbot", layout="centered")
st.title("🤖 Cosmos – Your ICH Guidelines Assistant")
st.markdown("Type your query below in English or Hinglish to explore ICH, CTD, dossier rules & more.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
Hi! I’m **Kailash Kothari**, the developer of this chatbot.  
This tool helps you quickly understand the right **ICH guidelines** for dossier preparation.

Whether it's **Q-Series**, **E-Series**, **eCTD**, or **Bioequivalence** — just ask and I’ll guide you.  
I’ve built this to simplify regulatory work with AI 🚀
""")

# Chat Clearing
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Chat input
if prompt := st.chat_input("Ask anything about ICH, dossier, eCTD..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    greeting = get_greeting()
    category = classify_query(prompt)
    local = search_guidelines(prompt)
    online = fetch_online_data(prompt)

    response = f"{greeting}\n\n**Category:** `{category}`\n\n**Guideline Info:**\n{local}\n\n**Wikipedia:**\n{online}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
