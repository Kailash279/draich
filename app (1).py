import streamlit as st
import json
import os
import requests
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename="app.log")
logger = logging.getLogger(__name__)

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="ICH Guidelines Chatbot", layout="centered")

# =============================
# Load BERT Model (Fallback safe)
# =============================
@st.cache_resource
def load_model():
    try:
        tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=5
        )
        logger.info("BERT model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load BERT model: {e}")
        st.error(f"Failed to load BERT model: {e}. Using fallback classification.")
        return None, None

with st.spinner("Loading BERT model..."):
    tokenizer, model = load_model()

def classify_query(text):
    if not tokenizer or not model:
        text = text.lower()
        if any(k in text for k in ["safety", "risk", "toxicology"]):
            return "Safety"
        elif any(k in text for k in ["quality", "manufacturing", "stability"]):
            return "Quality"
        elif any(k in text for k in ["efficacy", "clinical", "bioequivalence"]):
            return "Efficacy"
        elif any(k in text for k in ["general", "overview", "introduction"]):
            return "General"
        else:
            return "Miscellaneous"
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        categories = ["General", "Safety", "Quality", "Efficacy", "Miscellaneous"]
        return categories[torch.argmax(probs).item()]
    except Exception as e:
        logger.error(f"Query Classification Error: {e}")
        st.error(f"Query Classification Error: {e}")
        return "Miscellaneous"

# =============================
# Load Guidelines JSON
# =============================
def load_guideline_data():
    path = os.path.join(os.path.dirname(__file__), "ich_guidelines_full_combined.json")
    try:
        if not os.path.exists(path):
            st.error(f"JSON file not found at {path}. Please ensure the file exists.")
            logger.error(f"JSON file not found at {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info("JSON file loaded successfully")
            return data
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        st.error(f"Failed to load JSON file: {e}")
        return []

def search_guidelines(query):
    data = load_guideline_data()
    if not data:
        return "⚠️ No guidelines loaded due to JSON error."
    q = query.lower()
    results = []

    for g in data:
        if any(
            q in field.lower()
            for field in [g.get("title", ""), g.get("code", ""), g.get("purpose", ""), g.get("for_beginners", "")]
        ):
            subs = g.get("sub_guidelines", [])
            subs_text = "\n".join(f"- {s['code']}: {s['title']}" for s in subs) if subs else "None"

            results.append(
                f"""
### 📘 {g['code']} – {g['title']}
**Category:** {g['category']}  
**CTD Section:** {g['ctd_section']}  
**Introduced:** {g['introduced']}

🔍 **Purpose:** {g['purpose']}  
🧪 **Used For:** {g['used_for']}  
🧒 **Beginner Tip:** {g['for_beginners']}  
🔗 **Sub-Guidelines:**  
{subs_text}
"""
            )

    return "\n---\n".join(results) if results else "⚠️ No matching guidelines found."

# =============================
# Wiki Fallback Search
# =============================
def fetch_online_data(query):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            logger.info(f"Wikipedia API call successful for query: {query}")
            return r.json().get("extract", "No data found.")
        elif r.status_code == 404:
            logger.warning(f"Wikipedia article not found for query: {query}")
            return "🌐 No Wikipedia article found for this query."
        else:
            logger.error(f"Wikipedia API returned status {r.status_code} for query: {query}")
            return f"🌐 Wikipedia API returned status {r.status_code}."
    except requests.RequestException as e:
        logger.error(f"Failed to fetch Wikipedia data: {e}")
        st.error(f"Failed to fetch Wikipedia data: {e}")
        return "🌐 Unable to connect to Wikipedia."

# =============================
# Time-Based Greeting
# =============================
def get_greeting():
    h = datetime.now().hour
    return "Good Morning ☀️" if h < 12 else "Good Afternoon ☀️" if h < 18 else "Good Evening 🌙"

# =============================
# Hinglish Preprocessing
# =============================
def preprocess_query(query):
    hinglish_map = {
        "dawai": "medicine",
        "nirdesh": "guideline",
        "suraksha": "safety",
        "gunvatta": "quality",
        "prabhav": "efficacy"
    }
    query = query.lower()
    for hinglish, english in hinglish_map.items():
        query = query.replace(hinglish, english)
    return query

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
    st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Chat input
if prompt := st.chat_input("Ask anything about ICH, dossier, eCTD..."):
    processed_prompt = preprocess_query(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Searching guidelines..."):
        greeting = get_greeting()
        category = classify_query(processed_prompt)
        local = search_guidelines(processed_prompt)
        online = fetch_online_data(processed_prompt)

        response = f"{greeting}\n\n**Category:** `{category}`\n\n**Guideline Info:**\n{local}\n\n**Wikipedia:**\n{online}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)