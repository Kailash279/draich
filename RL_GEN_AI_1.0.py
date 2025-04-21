import json
import sqlite3
import difflib
import os
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
from PyPDF2 import PdfReader
import nltk

# =============================
# CONFIGURATION: SET DATA PATHS
# =============================

DATA_PATH = "D:/DATABASS/extracted_data.json"  # JSON Data Path
DB_PATH = "D:/DATABASS/pdf_data.db"  # SQLite Database Path
CHATBOT_MEMORY_PATH = "chatbot_memory.json"  # Reinforcement Learning Memory
PDF_FOLDER = "pdf_documents/"  # Folder for PDFs

# =============================
# LOAD NLP MODEL (BERT) WITH ERROR HANDLING
# =============================

def load_model():
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
        print("✅ Model Loaded Successfully")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Model Loading Error: {e}")
        return None, None

tokenizer, model = load_model()

if not tokenizer or not model:
    raise SystemExit("🚨 Model loading failed. Please check dependencies!")

# =============================
# DATABASE CONNECTION
# =============================

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS guidelines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    content TEXT
)
""")
conn.commit()

# =============================
# LOAD & STORE GUIDELINES DATA
# =============================

def load_json_data():
    """Load extracted JSON guidelines."""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def store_guidelines(data):
    """Store JSON extracted data into the database."""
    for entry in data:
        cursor.execute("INSERT INTO guidelines (filename, content) VALUES (?, ?)", 
                       (entry["filename"], entry["content"]))
    conn.commit()

# =============================
# SEARCH FUNCTIONS (DATABASE + JSON + PDF)
# =============================

def search_guidelines(query):
    """Search for relevant guidelines from the database & JSON."""
    cursor.execute("SELECT filename, content FROM guidelines WHERE content LIKE ?", (f"%{query}%",))
    db_results = cursor.fetchall()
    
    matched_results = [(filename, content[:300]) for filename, content in db_results]
    
    json_data = load_json_data()
    for entry in json_data:
        if query.lower() in entry["content"].lower():
            matched_results.append((entry["filename"], entry["content"][:300]))

    pdf_results = search_pdfs(query)
    matched_results.extend(pdf_results)
    
    return matched_results if matched_results else "⚠️ No relevant guideline found. Try different keywords."

def search_pdfs(query):
    """Search inside PDFs for relevant text."""
    matched_pdfs = []
    if os.path.exists(PDF_FOLDER):
        for pdf_file in os.listdir(PDF_FOLDER):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(PDF_FOLDER, pdf_file)
                with open(pdf_path, "rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text and query.lower() in text.lower():
                            matched_pdfs.append((pdf_file, text[:300]))
                            break  # Stop after first match
    return matched_pdfs

# =============================
# NLP MODEL: CLASSIFY USER QUERY
# =============================

def classify_query(user_input):
    """Use BERT to classify user query into predefined categories."""
    if not tokenizer or not model:
        return "Model loading error"
    
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs).logits
    probs = softmax(outputs, dim=1).detach().numpy()
    category = np.argmax(probs)
    
    categories = ["Safety & Toxicity", "Quality Guidelines", "Clinical Trials", "Regulatory Process", "General Query"]
    return categories[category]

# =============================
# REINFORCEMENT LEARNING (FEEDBACK LOOP)
# =============================

def update_response(query, response, feedback):
    """Update chatbot learning memory based on user feedback."""
    try:
        with open(CHATBOT_MEMORY_PATH, "r+") as file:
            data = json.load(file)
            if feedback == "positive":
                data[query] = response
            else:
                data[query] = "Retraining needed"
            file.seek(0)
            json.dump(data, file, indent=4)
    except FileNotFoundError:
        with open(CHATBOT_MEMORY_PATH, "w") as file:
            json.dump({query: response}, file, indent=4)

# =============================
# MAIN CHATBOT FUNCTION
# =============================

def chatbot():
    """Interactive chatbot with improved search functionality."""
    print("\n🔹 ICH RL Chatbot - Ask me about ICH guidelines! (Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        category = classify_query(user_input)
        results = search_guidelines(user_input)

        bot_response = f"🔎 **Category:** {category}\n\n{results}"
        print(f"\nBot: {bot_response}\n")

        feedback = input("Was this helpful? (yes/no): ").strip().lower()
        update_response(user_input, bot_response, "positive" if feedback in ["yes", "y"] else "negative")

# =============================
# EXECUTION (LOAD DATA & RUN CHATBOT)
# =============================

if __name__ == "__main__":
    json_data = load_json_data()
    store_guidelines(json_data)
    chatbot()
    conn.close()
