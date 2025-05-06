import streamlit as st
import json
import os
import requests
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

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
        model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        return tokenizer, model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None, None

tokenizer, model = load_model()

def classify_query(text):
    if not tokenizer or not model:
        return "Error loading model"
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        categories = ["General", "Safety", "Quality", "Efficacy", "Miscellaneous"]
        return categories[torch.argmax(probs).item()]
    except Exception as e:
        st.error(f"Query Classification Error: {e}")
        return "Error processing query"

# =============================
# Load Guidelines JSON
# =============================
def load_guideline_data():
    path = "ich_guidelines_full_combined.json"
    try:
        if not os.path.exists(path):
            st.error(f"JSON file not found at {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"JSON Load Error: {e}")
        return []

def search_guidelines(query):
    data = load_guideline_data()
    if not data:
        return "⚠️ No guidelines loaded due to JSON error."
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
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        r = requests.get(url)
        if r.status_code == 200:
            return r.json().get("extract", "No data found.")
        return "🌐 No Wikipedia article found."
    except Exception as e:
        st.error(f"Wikipedia Fetch Error: {e}")
        return "🌐 Error fetching Wikipedia data."

# =============================
# Time-Based Greeting
# =============================
def get_greeting():
    h = datetime.now().hour
    return "Good Morning ☀️" if h < 12 else "Good Afternoon ☀️" if h < 18 else "Good Evening 🌙"

# =============================
# Streamlit App UI
# =============================
st.title("🤖 Cosmos – Your ICH Guidelines Assistant")
st.markdown("Type your query below in English or Hinglish to explore ICH, CTD, dossier rules & more.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
**Developed by Kailash Kothari**  
Welcome to **Cosmos**, your go-to assistant for navigating **ICH guidelines** with ease!  

This AI-powered chatbot is designed to simplify regulatory processes, helping you prepare dossiers, understand **eCTD**, and apply **Q-Series**, **E-Series**, or **Bioequivalence** guidelines. Whether you're a beginner or a seasoned professional, Cosmos delivers clear, accurate answers to streamline your regulatory journey. 🚀  

Have questions? Just ask, and let’s master compliance together!
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