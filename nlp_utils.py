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

# Initialize NLP components
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    """
    Preprocess text by tokenizing, removing stopwords, and lemmatizing.
    """
    try:
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in stop_words
        ]
        
        return ' '.join(processed_tokens)
    except Exception as e:
        logging.error(f"Error in text preprocessing: {e}")
        return text.lower()

def extract_keywords(text, top_n=5):
    """
    Extract key phrases from text using spaCy.
    """
    try:
        doc = nlp(text)
        
        # Extract noun phrases and named entities
        keywords = []
        
        # Get noun phrases
        for chunk in doc.noun_chunks:
            keywords.append(chunk.text)
            
        # Get named entities
        for ent in doc.ents:
            keywords.append(ent.text)
            
        # Get important words (nouns, verbs, adjectives)
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and token.text.lower() not in stop_words:
                keywords.append(token.text)
        
        # Remove duplicates and return top N
        return list(dict.fromkeys(keywords))[:top_n]
    except Exception as e:
        logging.error(f"Error in keyword extraction: {e}")
        return []

def calculate_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts using sentence transformers.
    """
    try:
        # Encode texts
        embeddings1 = sentence_model.encode([text1])
        embeddings2 = sentence_model.encode([text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
        return float(similarity)
    except Exception as e:
        logging.error(f"Error in similarity calculation: {e}")
        return 0.0

def find_similar_guidelines(query, guidelines, threshold=0.3):
    """
    Find guidelines similar to the query using semantic similarity.
    """
    try:
        similar_guidelines = []
        
        for guideline in guidelines:
            # Combine relevant fields for comparison
            guideline_text = f"{guideline.get('title', '')} {guideline.get('purpose', '')} {guideline.get('for_beginners', '')}"
            
            # Calculate similarity
            similarity = calculate_similarity(query, guideline_text)
            
            if similarity > threshold:
                similar_guidelines.append((guideline, similarity))
        
        # Sort by similarity score
        similar_guidelines.sort(key=lambda x: x[1], reverse=True)
        return similar_guidelines
    except Exception as e:
        logging.error(f"Error in finding similar guidelines: {e}")
        return []

def analyze_query_intent(query):
    """
    Analyze the intent of the query using spaCy.
    """
    try:
        doc = nlp(query)
        
        # Extract key information
        intent = {
            'is_question': '?' in query,
            'main_verb': None,
            'entities': [],
            'key_topics': []
        }
        
        # Get main verb
        for token in doc:
            if token.pos_ == 'VERB':
                intent['main_verb'] = token.text
                break
        
        # Get named entities
        intent['entities'] = [ent.text for ent in doc.ents]
        
        # Get key topics (nouns)
        intent['key_topics'] = [
            token.text for token in doc
            if token.pos_ == 'NOUN' and token.text.lower() not in stop_words
        ]
        
        return intent
    except Exception as e:
        logging.error(f"Error in query intent analysis: {e}")
        return {
            'is_question': '?' in query,
            'main_verb': None,
            'entities': [],
            'key_topics': []
        } 