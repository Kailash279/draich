import json
import os
import re
import os
import json

MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def update_memory(question, answer):
    memory = load_memory()
    memory[question] = answer
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f)
def load_data():
    json_path = "ich_guidelines_full_combined.json"
    if not os.path.exists(json_path):
        return []

    with open(json_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
            return data if isinstance(data, list) else []
        except Exception as e:
            print("Error loading JSON:", e)
            return []

def clean_text(text):
    return re.sub(r'\s+', ' ', text.lower().strip())

def search_guidelines(query):
    data = load_data()
    results = []

    query = clean_text(query)
    for entry in data:
        content = clean_text(entry.get("content", ""))
        if query in content:
            results.append(entry.get("content", ""))

    return "\n\n".join(results) if results else "No matching guidelines found."
