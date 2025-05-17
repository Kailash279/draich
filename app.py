# app.py

import streamlit as st
import json
import os
import requests
from datetime import datetime
import logging
import sys
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# =========================
# ✅ Logging Setup
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_guidelines():
    try:
        with open("ich_guidelines_full_combined.json", "r", encoding="utf-8") as file:
            data = json.load(file)
            return data.get("guidelines", [])
    except Exception as e:
        logger.error(f"Error loading guidelines: {e}")
        return []

# =========================
# ✅ Load BERT Model
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=5
        ).to(device)
        
        logger.info("BERT model loaded successfully")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Failed to load BERT model: {e}")
        return None, None, None

# =========================
# ✅ Initialize model and guidelines
# =========================
tokenizer, model, device = load_model()
guidelines = load_guidelines()

def find_relevant_guidelines(query):
    query = query.lower()
    relevant = []
    
    for guideline in guidelines:
        # Search in multiple fields
        if (query in guideline.get("title", "").lower() or
            query in guideline.get("purpose", "").lower() or
            query in guideline.get("used_for", "").lower() or
            query in guideline.get("for_beginners", "").lower()):
            relevant.append(guideline)
    
    return relevant

# =========================
# ✅ Classify Query
# =========================
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
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = softmax(outputs.logits, dim=1)
        categories = ["General", "Safety", "Quality", "Efficacy", "Miscellaneous"]
        return categories[torch.argmax(probs).item()]
    except Exception as e:
        logger.error(f"Query Classification Error: {e}")
        return "Miscellaneous"

# =========================
# ✅ Streamlit App UI
# =========================
st.set_page_config(page_title="ICH Guidelines Assistant", page_icon="📘")
st.title("ICH Guidelines Assistant 🤖")
st.write("Hi, I am **Kailash Kothari**. Welcome to the ICH Guidelines Assistant. Ask any question about ICH guidelines!")

# =========================
# ✅ Chat Memory Init
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# =========================
# ✅ Chat History Display
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# ✅ Chat Input Area
# =========================
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        padding: 0;
        margin-left: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])

with col1:
    st.session_state.user_input = st.text_input("Type your question here...", key="input_box", label_visibility="collapsed", value=st.session_state.user_input)

with col2:
    if st.button("➤", key="send_button"):
        user_query = st.session_state.user_input.strip()
        if user_query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Find relevant guidelines
            relevant_guidelines = find_relevant_guidelines(user_query)
            
            # Classify the query
            category = classify_query(user_query)
            
            # Prepare response
            if relevant_guidelines:
                response = f"**Category:** {category}\n\n**Relevant Guidelines:**\n\n"
                for guideline in relevant_guidelines:
                    response += f"### 📘 {guideline['code']} - {guideline['title']}\n"
                    response += f"**Purpose:** {guideline['purpose']}\n"
                    response += f"**Used For:** {guideline['used_for']}\n"
                    response += f"**Beginner Tip:** {guideline['for_beginners']}\n\n"
            else:
                response = f"**Category:** {category}\n\nI couldn't find specific guidelines matching your query. Please try rephrasing or asking about a different aspect of ICH guidelines."
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Clear the input
            st.session_state.user_input = ""
            
            # Rerun to update the chat
            st.experimental_rerun()
        else:
            st.warning("Please enter a question!")

# =========================
# ✅ Sidebar - About
# =========================
with st.sidebar:
    st.header("📘 About This App")
    st.markdown("""
**ICH Guidelines Assistant 🤖**  
Built with ❤️ by **Kailash Kothari**

💡 This assistant helps you understand and classify topics from ICH Guidelines, including:

- 🧪 **Quality Guidelines**  
- 🔬 **Safety & Toxicology**  
- 💊 **Efficacy & Clinical Trials**  
- 📂 **General Regulatory Concepts**

Whether you're working in regulatory affairs, pharmaceutical R&D, or quality assurance — this tool is designed to support your compliance and learning journey.
    """)
