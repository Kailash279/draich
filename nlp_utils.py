import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Ensure required NLTK data is downloaded
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

# Initialize NLP tools
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize the text."""
    try:
        tokens = word_tokenize(text.lower())
        processed_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in stop_words
        ]
        return ' '.join(processed_tokens)
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return text.lower()

def extract_keywords(text, top_n=5):
    """Extract important phrases and keywords from text."""
    try:
        doc = nlp(text)
        keywords = []

        # Noun chunks
        keywords += [chunk.text for chunk in doc.noun_chunks]

        # Named entities
        keywords += [ent.text for ent in doc.ents]

        # Important words (NOUN, VERB, ADJ)
        keywords += [
            token.text for token in doc
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and token.text.lower() not in stop_words
        ]

        # Remove duplicates while preserving order
        return list(dict.fromkeys(keywords))[:top_n]
    except Exception as e:
        logging.error(f"Keyword extraction error: {e}")
        return []

def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts."""
    try:
        embeddings1 = sentence_model.encode([text1])
        embeddings2 = sentence_model.encode([text2])
        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
        return float(similarity)
    except Exception as e:
        logging.error(f"Similarity calculation error: {e}")
        return 0.0

def find_similar_guidelines(query, guidelines, threshold=0.3):
    """Find similar guidelines from a list based on query."""
    try:
        similar_guidelines = []
        for guideline in guidelines:
            guideline_text = f"{guideline.get('title', '')} {guideline.get('purpose', '')} {guideline.get('for_beginners', '')}"
            similarity = calculate_similarity(query, guideline_text)
            if similarity > threshold:
                similar_guidelines.append((guideline, similarity))
        similar_guidelines.sort(key=lambda x: x[1], reverse=True)
        return similar_guidelines
    except Exception as e:
        logging.error(f"Guideline similarity error: {e}")
        return []

def analyze_query_intent(query):
    """Analyze query intent and extract useful semantic info."""
    try:
        doc = nlp(query)
        intent = {
            'is_question': '?' in query,
            'main_verb': None,
            'entities': [ent.text for ent in doc.ents],
            'key_topics': [
                token.text for token in doc
                if token.pos_ == 'NOUN' and token.text.lower() not in stop_words
            ]
        }

        for token in doc:
            if token.pos_ == 'VERB':
                intent['main_verb'] = token.text
                break

        return intent
    except Exception as e:
        logging.error(f"Intent analysis error: {e}")
        return {
            'is_question': '?' in query,
            'main_verb': None,
            'entities': [],
            'key_topics': []
        }
