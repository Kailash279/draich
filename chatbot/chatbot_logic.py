from django.conf import settings
import json
import sqlite3
import difflib
import os
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
from PyPDF2 import PdfReader

# Initialize BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# Database connection
conn = sqlite3.connect(settings.DB_PATH)
cursor = conn.cursor()

# Create necessary functions from original file
def search_guidelines(query):
    """Search for relevant guidelines from the database & JSON."""
    
    # Fetch all data from database
    cursor.execute("SELECT filename, content FROM guidelines")
    db_results = cursor.fetchall()

    # Fuzzy matching for text similarity
    matched_results = []
    for filename, content in db_results:
        if query.lower() in content.lower():
            matched_results.append((filename, content[:300]))  # Show first 300 characters

    # JSON data search
    json_data = load_json_data()
    for entry in json_data:
        if query.lower() in entry["content"].lower():
            matched_results.append((entry["filename"], entry["content"][:300]))

    # PDF search
    pdf_results = search_pdfs(query)
    matched_results.extend(pdf_results)

    # Prepare final response
    if matched_results:
        response = "\n".join([f"📄 {row[0]}:\n{row[1]}..." for row in matched_results[:3]])  # Show top 3 results
    else:
        response = "⚠️ No relevant guideline found. Try different keywords."

    return response

def classify_query(user_input):
    """Use BERT to classify user query into predefined categories."""
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs).logits
    probs = softmax(outputs, dim=1).detach().numpy()
    category = np.argmax(probs)
    
    categories = ["Safety & Toxicity", "Quality Guidelines", "Clinical Trials", "Regulatory Process", "General Query"]
    return categories[category]

def update_response(query, response, feedback):
    """Update chatbot learning memory based on user feedback."""
    try:
        with open(settings.CHATBOT_MEMORY_PATH, "r+") as file:
            data = json.load(file)
            if feedback == "positive":
                data[query] = response  # Reinforce correct response
            else:
                data[query] = "Retraining needed"  # Mark for review
            file.seek(0)
            json.dump(data, file, indent=4)
    except FileNotFoundError:
        with open(settings.CHATBOT_MEMORY_PATH, "w") as file:
            json.dump({query: response}, file, indent=4)

def load_json_data():
    """Load extracted JSON guidelines."""
    try:
        with open(settings.DATA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def store_guidelines(data):
    """Store JSON extracted data into the database."""
    for entry in data:
        cursor.execute("INSERT INTO guidelines (filename, content) VALUES (?, ?)", 
                       (entry["filename"], entry["content"]))
    conn.commit()

def search_pdfs(query):
    """Search inside PDFs for relevant text."""
    matched_pdfs = []
    if os.path.exists(settings.PDF_FOLDER):
        for pdf_file in os.listdir(settings.PDF_FOLDER):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(settings.PDF_FOLDER, pdf_file)
                with open(pdf_path, "rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text and query.lower() in text.lower():
                            matched_pdfs.append((pdf_file, text[:300]))  # First 300 characters
                            break  # Stop after first match
    return matched_pdfs