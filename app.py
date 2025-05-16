
import json
import os
import requests
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
import logging
import sys
from nlp_utils import (
    preprocess_text,
    extract_keywords,
    find_similar_guidelines,
    analyze_query_intent
)

def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()

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

# Load BERT Model
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

# Initialize model
tokenizer, model, device = load_model()

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

def greet(name):
    return "Hello " + name + "!!"

# Create Gradio interface
demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="ICH Guidelines Assistant",
    description=" Hiii I AM Kailash Kothari, Welcome to the ICH Guidelines Assistant. Ask any questions about ICH guidelines."
)

if __name__ == "__main__":
    demo.launch()

    