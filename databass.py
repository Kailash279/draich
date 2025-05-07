import json
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename="app.log")
logger = logging.getLogger(__name__)

# Note: Consider renaming this file to 'database.py' for clarity
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")

def load_memory():
    """
    Load the memory JSON file containing question-answer pairs.
    Returns an empty dict if the file doesn't exist or is invalid.
    """
    try:
        if not os.path.exists(MEMORY_FILE):
            logger.info(f"Memory file not found at {MEMORY_FILE}. Returning empty dict.")
            return {}
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                logger.error(f"Invalid memory file format: {MEMORY_FILE}")
                return {}
            logger.info("Memory file loaded successfully")
            return data
    except Exception as e:
        logger.error(f"Failed to load memory file: {e}")
        return {}

def update_memory(question, answer):
    """
    Update the memory JSON file with a new question-answer pair.
    Validates inputs and handles file write errors.
    """
    if not isinstance(question, str) or not question.strip():
        logger.error("Invalid question: must be a non-empty string")
        return
    if not isinstance(answer, str):
        logger.error("Invalid answer: must be a string")
        return

    try:
        memory = load_memory()
        memory[question.strip()] = answer.strip()
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
        logger.info(f"Memory updated with question: {question}")
    except Exception as e:
        logger.error(f"Failed to update memory file: {e}")

def load_data():
    """
    Load the ICH guidelines JSON file.
    Returns an empty list if the file is missing or invalid.
    """
    json_path = os.path.join(os.path.dirname(__file__), "ich_guidelines_full_combined.json")
    try:
        if not os.path.exists(json_path):
            logger.error(f"JSON file not found at {json_path}")
            return []
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if not isinstance(data, list):
                logger.error(f"Invalid JSON format: {json_path} is not a list")
                return []
            # Basic validation: ensure each entry has required fields
            for entry in data:
                if not isinstance(entry, dict) or "content" not in entry:
                    logger.warning(f"Invalid entry in JSON: {entry}")
                    return []
            logger.info("Guidelines JSON loaded successfully")
            return data
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return []

def clean_text(text):
    """
    Clean and normalize text by removing extra whitespace and converting to lowercase.
    Returns an empty string if input is invalid.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text.lower().strip())

def search_guidelines(query):
    """
    Search ICH guidelines for entries matching the query in multiple fields.
    Returns formatted results or a message if no matches are found.
    """
    data = load_data()
    if not data:
        return "⚠️ No guidelines loaded due to JSON error."

    query = clean_text(query)
    if not query:
        return "⚠️ Invalid query provided."

    results = []
    for entry in data:
        # Search in multiple fields for better coverage
        fields = [
            entry.get("content", ""),
            entry.get("title", ""),
            entry.get("code", ""),
            entry.get("purpose", ""),
            entry.get("for_beginners", "")
        ]
        if any(query in clean_text(field) for field in fields):
            # Format result for consistency with app.py
            subs = entry.get("sub_guidelines", [])
            subs_text = "\n".join(f"- {s['code']}: {s['title']}" for s in subs) if subs else "None"
            result = f"""
### 📘 {entry.get('code', 'N/A')} – {entry.get('title', 'Untitled')}
**Category:** {entry.get('category', 'N/A')}  
**CTD Section:** {entry.get('ctd_section', 'N/A')}  
**Introduced:** {entry.get('introduced', 'N/A')}

🔍 **Purpose:** {entry.get('purpose', 'N/A')}  
🧪 **Used For:** {entry.get('used_for', 'N/A')}  
🧒 **Beginner Tip:** {entry.get('for_beginners', 'N/A')}  
🔗 **Sub-Guidelines:**  
{subs_text}
"""
            results.append(result)

    return "\n---\n".join(results) if results else "⚠️ No matching guidelines found."