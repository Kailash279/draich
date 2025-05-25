import streamlit as st
import json
import logging
import sys
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load guidelines JSON file
def load_guidelines():
    try:
        with open("ich_guidelines_full_combined.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("guidelines", [])
    except Exception as e:
        logger.error(f"Failed to load guidelines: {e}")
        return []

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

        logger.info("Model loaded successfully")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return None, None, None

# Classify query using BERT or fallback keyword method
def classify_query(text, tokenizer, model, device):
    if not tokenizer or not model:
        # Fallback basic keyword matching
        text_lower = text.lower()
        if any(k in text_lower for k in ["safety", "risk", "toxicology"]):
            return "Safety"
        elif any(k in text_lower for k in ["quality", "manufacturing", "stability"]):
            return "Quality"
        elif any(k in text_lower for k in ["efficacy", "clinical", "bioequivalence"]):
            return "Efficacy"
        elif any(k in text_lower for k in ["general", "overview", "introduction"]):
            return "General"
        else:
            return "Miscellaneous"
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        categories = ["General", "Safety", "Quality", "Efficacy", "Miscellaneous"]
        pred = torch.argmax(probs).item()
        return categories[pred]
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return "Miscellaneous"

# Find guidelines matching query
def find_relevant_guidelines(query, guidelines):
    query_lower = query.lower()
    relevant = []
    for g in guidelines:
        if (query_lower in g.get("title", "").lower() or
            query_lower in g.get("purpose", "").lower() or
            query_lower in g.get("used_for", "").lower() or
            query_lower in g.get("for_beginners", "").lower()):
            relevant.append(g)
    return relevant

# Load once
tokenizer, model, device = load_model()
guidelines = load_guidelines()

# Streamlit UI setup
st.set_page_config(page_title="ICH Guidelines Assistant", page_icon="📘")
st.title("ICH Guidelines Assistant \ud83e\udd16")
st.write("Hi, I am **Kailash Kothari**. Ask me anything about ICH guidelines!")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

col1, col2 = st.columns([6, 1])

with col1:
    st.session_state.user_input = st.text_input("Type your question here...", key="input_box", value=st.session_state.user_input, label_visibility="collapsed")

with col2:
    if st.button("\u27a4", key="send_button"):
        query = st.session_state.user_input.strip()
        if query:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})

            # Find guidelines & classify
            relevant = find_relevant_guidelines(query, guidelines)
            category = classify_query(query, tokenizer, model, device)

            if relevant:
                response = f"**Category:** {category}\n\n**Relevant Guidelines:**\n\n"
                for g in relevant:
                    response += f"### \ud83d\udcd8 {g['code']} - {g['title']}\n"
                    response += f"**Purpose:** {g['purpose']}\n"
                    response += f"**Used For:** {g['used_for']}\n"
                    response += f"**Beginner Tip:** {g['for_beginners']}\n\n"
            else:
                response = f"**Category:** {category}\n\nNo exact guidelines found. Please try rephrasing or ask something else."

            # Add assistant reply
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Clear input and rerun
            st.session_state.user_input = ""
            st.experimental_rerun()
        else:
            st.warning("Please enter a question!")

# Sidebar info
with st.sidebar:
    st.header("About This App")
    st.markdown("""
**ICH Guidelines Assistant 🤖**  
Created by **Kailash Kothari**

This app helps you explore and classify ICH guidelines topics such as Quality, Safety, Efficacy, and more.
""")
