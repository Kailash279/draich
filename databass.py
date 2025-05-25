import json
import os

DATA_PATH = "ich_guidelines_database.json"
MEMORY_PATH = "memory.json"

# Load ICH guideline data
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Search ICH guidelines
def search_guidelines(query, data):
    query_lower = query.lower()
    response = []

    for category, info in data.get("guidelines", {}).items():
        for code, g in info.get("guidelines", {}).items():
            combined = f"{code} {g.get('name', '')} {g.get('purpose', '')} {g.get('use', '')}"
            if query_lower in combined.lower():
                r = f"📘 **{code} - {g['name']}**\nPurpose: {g['purpose']}\nUse: {g['use']}\n"
                if 'sub_guidelines' in g:
                    for sub_code, sub_title in g['sub_guidelines'].items():
                        r += f"   - {sub_code}: {sub_title}\n"
                response.append(r)
    return "\n\n".join(response[:5]) if response else ""

# Update memory
def update_memory(query, response):
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            memory = json.load(f)
    else:
        memory = {}
    memory[query.lower()] = response
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

# ✅ INSERT THIS BELOW
def load_memory(query):
    if not os.path.exists(MEMORY_PATH):
        return ""
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        memory = json.load(f)
    return memory.get(query.lower(), "")


def search_guidelines(query, data):
    query_lower = query.lower()
    response = []

    for category, info in data.get("guidelines", {}).items():
        for code, g in info.get("guidelines", {}).items():
            combined = f"{code} {g.get('name', '')} {g.get('purpose', '')} {g.get('use', '')}"
            if query_lower in combined.lower():
                r = f"📘 **{code} - {g['name']}**\nPurpose: {g['purpose']}\nUse: {g['use']}\n"
                response.append(r)

    return "\n\n".join(response[:5]) if response else "Sorry, I couldn't find that in guidelines."
