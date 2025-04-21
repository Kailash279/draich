import json
import os
import re

def load_data():
    json_path = "restructured_guidelines.json"
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
