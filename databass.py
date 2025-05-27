# databass.py
import json
import os
import fitz  # PyMuPDF for PDF
import pytesseract  # For OCR
from PIL import Image

DATA_PATH = "ich_guidelines_database.json"
MEMORY_PATH = "memory.json"

# ================================
# 📘 Load ICH Guidelines
# ================================
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ================================
# 🔍 Search guidelines
# ================================
def search_guidelines(query, data):
    query_lower = query.lower()
    results = []

    for category, section in data.get("guidelines", {}).items():
        if query_lower in category.lower():
            results.append(f"📘 **{category} Guidelines**\n{section.get('description', '')}\n")

        for code, detail in section.get("guidelines", {}).items():
            full_text = f"{code} {detail.get('name', '')} {detail.get('purpose', '')} {detail.get('use', '')}"
            if query_lower in full_text.lower():
                entry = f"📘 **{code} - {detail['name']}**\nPurpose: {detail['purpose']}\nUse: {detail['use']}\nLast Updated: {detail.get('last_updated', 'N/A')}"
                if 'sub_guidelines' in detail:
                    entry += "\n\nSub-Guidelines:\n"
                    for sub_code, sub_desc in detail['sub_guidelines'].items():
                        entry += f"🔹 {sub_code}: {sub_desc}\n"
                results.append(entry)

    return "\n\n".join(results) if results else "Sorry, no matching guideline found."

# ================================
# 📂 Memory: load & update
# ================================
def update_memory(query, response):
    if not response or response.strip().lower() in ["none", "null", ""]:
        return  # skip invalid entries

    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            memory = json.load(f)
    else:
        memory = {}

    memory[query.lower()] = response
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def load_memory(query):
    if not os.path.exists(MEMORY_PATH):
        return ""
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        memory = json.load(f)
    return memory.get(query.lower(), "")

# ================================
# 🌐 Overview Generator
# ================================
def get_ich_overview(data):
    ov = data.get("overview", {})
    return f"""
**🌐 What is ICH?**

{ov.get('what_is_ich', '')}

📅 **Established**: {ov.get('established', 'N/A')}  
🎯 **Mission**: {ov.get('mission', 'N/A')}  
👥 **Members**: {', '.join(ov.get('members', []))}  
👀 **Observers**: {', '.join(ov.get('observers', []))}
"""

# ================================
# 📤 Learn from Uploaded File
# ================================
def learn_from_text(uploaded_file):
    try:
        if uploaded_file.name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8")

        elif uploaded_file.name.endswith(".pdf"):
            text = ""
            pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in pdf:
                text += page.get_text()
            return text.strip()

        elif uploaded_file.name.endswith((".jpg", "jpeg", "png")):
            image = Image.open(uploaded_file)
            return pytesseract.image_to_string(image)

        else:
            return ""

    except Exception as e:
        return f"Error reading file: {e}"
def learn_from_text(uploaded_file):
    try:
        # Handle .txt files
        if uploaded_file.name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8")

        # Handle .pdf files
        elif uploaded_file.name.endswith(".pdf"):
            import fitz
            text = ""
            pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in pdf:
                text += page.get_text()
            return text.strip()

        # ❌ Image OCR skipped on cloud
        elif uploaded_file.name.endswith(("jpg", "jpeg", "png")):
            return "Image upload is not supported in cloud version."

        else:
            return ""
    except Exception as e:
        return f"Error reading file: {e}"
