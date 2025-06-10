# nlp_utils.py

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# NLP models
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==================================
# 🔹 Rule-based Query Classification
# ==================================
def classify_query(text):
    text = text.lower()

    if any(word in text for word in ["hello", "hi", "namaste"]):
        return "Greeting"
    elif any(word in text for word in ["bye", "goodbye", "alvida"]):
        return "Farewell"
    elif any(word in text for word in ["madad", "help", "support", "query"]):
        return "Support Query"
    elif any(word in text for word in ["formulation", "banane", "development", "pdr"]):
        return "Product Development"
    elif any(word in text for word in ["stability", "shelf life", "q1"]):
        return "Stability Testing"
    elif any(word in text for word in ["validation", "analytical", "test", "method"]):
        return "Analytical Validation"
    else:
        return "General Query"

# ==================================
# 🔹 Text Preprocessing
# ==================================
def preprocess_text(text):
    try:
        tokens = word_tokenize(text.lower())
        return " ".join([
            lemmatizer.lemmatize(t) for t in tokens
            if t.isalnum() and t not in stop_words
        ])
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return text.lower()

# ==================================
# 🔹 Keyword + Phrase Extraction
# ==================================
def extract_keywords(text, top_n=5):
    try:
        doc = nlp(text)
        keywords = []

        keywords += [chunk.text for chunk in doc.noun_chunks]
        keywords += [ent.text for ent in doc.ents]
        keywords += [
            token.text for token in doc
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and token.text.lower() not in stop_words
        ]
        return list(dict.fromkeys(keywords))[:top_n]
    except Exception as e:
        logging.error(f"Keyword extraction error: {e}")
        return []

# ==================================
# 🔹 Semantic Similarity (BERT)
# ==================================
def calculate_similarity(text1, text2):
    try:
        emb1 = sentence_model.encode([text1])
        emb2 = sentence_model.encode([text2])
        return float(cosine_similarity(emb1, emb2)[0][0])
    except Exception as e:
        logging.error(f"Similarity error: {e}")
        return 0.0

def find_similar_guidelines(query, guidelines, threshold=0.3):
    try:
        similar = []
        for g in guidelines:
            combined = f"{g.get('title', '')} {g.get('purpose', '')} {g.get('for_beginners', '')}"
            sim_score = calculate_similarity(query, combined)
            if sim_score > threshold:
                similar.append((g, sim_score))
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    except Exception as e:
        logging.error(f"Guideline match error: {e}")
        return []

# ==================================
# 🔹 Query Intent Analyzer
# ==================================
def analyze_query_intent(query):
    try:
        doc = nlp(query)
        return {
            "is_question": "?" in query,
            "main_verb": next((t.text for t in doc if t.pos_ == "VERB"), None),
            "entities": [ent.text for ent in doc.ents],
            "key_topics": [
                t.text for t in doc if t.pos_ == "NOUN" and t.text.lower() not in stop_words
            ]
        }
    except Exception as e:
        logging.error(f"Intent analysis error: {e}")
        return {
            "is_question": "?" in query,
            "main_verb": None,
            "entities": [],
            "key_topics": []
        }
