from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)

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

# =============================
# Load Guidelines JSON
# =============================
@st.cache_data
def load_guideline_data():
    try:
        # Use relative path for Streamlit Cloud
        path = "ich_guidelines_full_combined.json"
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
    
    # Check if data has the correct structure
    if "guidelines" not in data:
        logger.error("JSON data does not contain 'guidelines' key")
        return "⚠️ Invalid data structure in guidelines file."
    
    # Preprocess the query
    processed_query = preprocess_text(query)
    
    # Extract keywords from the query
    keywords = extract_keywords(query)
    
    # Analyze query intent
    intent = analyze_query_intent(query)
    
    # Find similar guidelines using semantic similarity
    similar_guidelines = find_similar_guidelines(processed_query, data["guidelines"])
    
    results = []
    
    # Add keyword-based results
    for g in data["guidelines"]:
        try:
            # Get all searchable fields with safe defaults
            searchable_fields = [
                str(g.get("title", "")),
                str(g.get("code", "")),
                str(g.get("purpose", "")),
                str(g.get("for_beginners", ""))
            ]
            
            # Check if any keyword matches
            if any(keyword.lower() in ' '.join(searchable_fields).lower() for keyword in keywords):
                subs = g.get("sub_guidelines", [])
                subs_text = "\n".join(f"- {s.get('code', 'N/A')}: {s.get('title', 'N/A')}" for s in subs) if subs else "None"

                results.append(
                    f"""
### 📘 {g.get('code', 'N/A')} – {g.get('title', 'N/A')}
**Category:** {g.get('category', 'N/A')}  
**CTD Section:** {g.get('ctd_section', 'N/A')}  
**Introduced:** {g.get('introduced', 'N/A')}

🔍 **Purpose:** {g.get('purpose', 'N/A')}  
🧪 **Used For:** {g.get('used_for', 'N/A')}  
🧒 **Beginner Tip:** {g.get('for_beginners', 'N/A')}  
🔗 **Sub-Guidelines:**  
{subs_text}
"""
                )
        except Exception as e:
            logger.error(f"Error processing guideline entry: {e}")
            continue
    
    # Add semantically similar results
    for guideline, similarity in similar_guidelines:
        if similarity > 0.3:  # Only include highly similar results
            g = guideline
            subs = g.get("sub_guidelines", [])
            subs_text = "\n".join(f"- {s.get('code', 'N/A')}: {s.get('title', 'N/A')}" for s in subs) if subs else "None"
            
            results.append(
                f"""
### 📘 {g.get('code', 'N/A')} – {g.get('title', 'N/A')}
**Category:** {g.get('category', 'N/A')}  
**CTD Section:** {g.get('ctd_section', 'N/A')}  
**Introduced:** {g.get('introduced', 'N/A')}

🔍 **Purpose:** {g.get('purpose', 'N/A')}  
🧪 **Used For:** {g.get('used_for', 'N/A')}  
🧒 **Beginner Tip:** {g.get('for_beginners', 'N/A')}  
🔗 **Sub-Guidelines:**  
{subs_text}

*Similarity Score: {similarity:.2f}*
"""
            )
    
    # Add query analysis information
    analysis_info = f"""
### 🔍 Query Analysis
**Keywords:** {', '.join(keywords)}
**Intent:** {'Question' if intent['is_question'] else 'Statement'}
**Main Topics:** {', '.join(intent['key_topics'])}
"""
    
    return analysis_info + "\n---\n" + "\n---\n".join(results) if results else "⚠️ No matching guidelines found."

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        category = classify_query(query)
        response = search_guidelines(query)
        return jsonify({
            'category': category,
            'response': response
        })
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))

    