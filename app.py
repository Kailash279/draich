import streamlit as st
import json
import sqlite3
import os
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
from PyPDF2 import PdfReader

# Configuration
DATA_PATH = os.getenv('DATA_PATH', "D:/DATABASS/extracted_data.json")
DB_PATH = os.getenv('DB_PATH', "D:/DATABASS/pdf_data.db")
CHATBOT_MEMORY_PATH = os.getenv('CHATBOT_MEMORY_PATH', "chatbot_memory.json")
PDF_FOLDER = os.getenv('PDF_FOLDER', "pdf_documents/")

# Create necessary directories if they don't exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

# Initialize BERT model with error handling
@st.cache_resource
def load_model():
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

tokenizer, model = load_model()

# Database connection with error handling
try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS guidelines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        content TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subscribers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
except sqlite3.Error as e:
    st.error(f"Database error: {str(e)}")
    conn = None
    cursor = None

def search_guidelines(query):
    try:
        # Basic search implementation
        if not cursor:
            return "Database connection error"
            
        cursor.execute("SELECT content FROM guidelines WHERE content LIKE ?", (f"%{query}%",))
        results = cursor.fetchall()
        
        if not results:
            return "No matching guidelines found."
            
        return "\n".join([result[0] for result in results])
    except Exception as e:
        return f"Error searching guidelines: {str(e)}"

def classify_query(user_input):
    try:
        if not tokenizer or not model:
            return "Model loading error"
            
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs).item()
        
        # Map class index to category name (customize based on your categories)
        categories = ["General", "Safety", "Quality", "Efficacy", "Other"]
        return categories[predicted_class]
    except Exception as e:
        return f"Classification error: {str(e)}"

def update_response(query, response, feedback):
    try:
        if not os.path.exists(CHATBOT_MEMORY_PATH):
            memory = {}
        else:
            with open(CHATBOT_MEMORY_PATH, 'r') as f:
                memory = json.load(f)
        
        if query not in memory:
            memory[query] = {"response": response, "feedback": []}
        
        memory[query]["feedback"].append(feedback)
        
        with open(CHATBOT_MEMORY_PATH, 'w') as f:
            json.dump(memory, f)
    except Exception as e:
        st.error(f"Error updating response: {str(e)}")

def save_email(email):
    try:
        if not cursor:
            return False, "Database connection error"
        cursor.execute("INSERT INTO subscribers (email) VALUES (?)", (email,))
        conn.commit()
        return True, "Email successfully saved!"
    except sqlite3.IntegrityError:
        return False, "Email already registered!"
    except Exception as e:
        return False, f"Error saving email: {str(e)}"

# Streamlit UI
st.title("ICH Guidelines Chatbot 🤖")
st.write("Ask me anything about ICH guidelines!")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get bot response
    category = classify_query(prompt)
    results = search_guidelines(prompt)
    
    response = f"**Category:** {category}\n\n{results}"
    
    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

    # Feedback buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Helpful"):
            update_response(prompt, response, "positive")
            st.success("Thanks for the feedback!")
    with col2:
        if st.button("👎 Not Helpful"):
            update_response(prompt, response, "negative")
            st.error("Sorry about that. We'll try to improve!")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write("Hi, I am Kailash Kothari, the developer of this chatbot. I need your help in training this bot It’s not fully ready yet, but with your assistance, it will start working soon. With your support, this bot can improve and become even better!")
    st.divider()
    
    # Email subscription form
    st.header("Stay Updated!")
    st.write("Receive helpful onboarding emails, news, and occasional swag!")
    email = st.text_input("Email:", key="emai_input")
    
    if email:
        if not '@' in email or not '.' in email:
            st.error("Please enter a valid email address")
        elif st.button("Subscribe"):
            success, message = save_email(email)
            if success:
                st.success(message)
                st.session_state.email_input = ""  # Clear the input
            else:
                st.error(message)
    
    st.divider()
    st.write("Using:")
    st.write("- BERT for classification")
    st.write("- SQLite for storage")
    st.write("- PDF and JSON data sources")
