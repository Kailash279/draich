import streamlit as st
import json
import os
import requests
from datetime import datetime
import logging
import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ======================
# Load BERT Model
# ======================
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=5  # Just assumed — not trained!
        ).to(device)
        
        logger.info("✅ BERT model loaded")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"❌ BERT Load Failed: {e}")
        return None, None, None

tokenizer, model, device = load_model()

# ======================
# Classify Query Function
# ======================
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
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        categories = ["General", "Safety", "Quality", "Efficacy", "Miscellaneous"]
        return categories[torch.argmax(probs).item()]
    except Exception as e:
        logger.error(f"❌ Classification Failed: {e}")
        return "Miscellaneous"

# ======================
# Greeting
# ======================
def greet(name):
    return f"Hello {name}!!"

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="ICH Guidelines Assistant", page_icon="📘")

st.title("ICH Guidelines Assistant 🤖")
st.write("👋 Hi, I'm **Kailash Kothari's Assistant** for ICH Guidelines. Ask anything related to Safety, Quality, Efficacy, and dossier preparation.")

name = st.text_input("Enter your name")

if st.button("Greet"):
    if name:
        st.success(greet(name))
    else:
        st.warning("Please enter your name.")
