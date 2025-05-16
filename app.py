# app.py

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from datetime import datetime
import logging
import sys

# =========================
# ✅ Logging Setup
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =========================
# ✅ Load BERT Model
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=5
        ).to(device)
        logger.info("✅ Model loaded on %s", device)
        return tokenizer, model, device
    except Exception as e:
        logger.error("❌ Model loading failed: %s", e)
        return None, None, None

tokenizer, model, device = load_model()

# =========================
# ✅ Classify User Input
# =========================
def classify_query(text):
    categories = ["General", "Safety", "Quality", "Efficacy", "Miscellaneous"]
    
    if not tokenizer or not model:
        text = text.lower()
        if "safety" in text or "toxic" in text:
            return "Safety"
        elif "quality" in text or "stability" in text:
            return "Quality"
        elif "efficacy" in text or "clinical" in text:
            return "Efficacy"
        elif "overview" in text or "intro" in text:
            return "General"
        return "Miscellaneous"

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        return categories[torch.argmax(probs).item()]
    except Exception as e:
        logger.error("❌ Classification failed: %s", e)
        return "Miscellaneous"

# =========================
# ✅ Greeting
# =========================
def greet(name):
    return f"Hello {name}!! 👋 Welcome to ICH Guidelines Assistant."

# =========================
# ✅ Streamlit App UI
# =========================
st.set_page_config(page_title="ICH Guidelines Assistant", layout="centered", page_icon="📘")
st.title("ICH Guidelines Assistant 🤖")
st.write("Ask your questions related to **ICH Guidelines** — Quality, Safety, Efficacy, etc.")

# User Name Input
name = st.text_input("Enter your name")

if st.button("Greet Me"):
    if name:
        st.success(greet(name))
    else:
        st.warning("Please enter your name first.")

# User Query
user_input = st.text_input("🔎 Type your ICH query here...")

if user_input:
    st.info("⏳ Classifying your query...")
    category = classify_query(user_input)
    st.success(f"📂 Predicted Category: **{category}**")

# =========================
# ✅ Sidebar - About Section
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
