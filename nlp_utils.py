from nlp_utils import find_similar_guidelines  # Add this import

def search_guidelines(query, data):
    query_lower = query.lower()
    results = []

    # -------- Step 1: Exact Matching (same as before) --------
    for category, section in data.get("guidelines", {}).items():
        if query_lower in category.lower():
            results.append(f"📘 **{category} Guidelines**\n{section.get('description', '')}\n")

        for code, detail in section.get("guidelines", {}).items():
            full_text = f"{code} {detail.get('name', '')} {detail.get('purpose', '')} {detail.get('use', '')}"
            if query_lower in full_text.lower():
                entry = f"📘 **{code} - {detail['name']}**\nPurpose: {detail['purpose']}\nUse: {detail['use']}\n"
                results.append(entry)

    if results:
        return "\n\n".join(results)

    # -------- Step 2: Fallback NLP Matching if no exact match --------
    all_guidelines = []
    for category, section in data.get("guidelines", {}).items():
        for code, detail in section.get("guidelines", {}).items():
            all_guidelines.append({
                "code": code,
                "title": detail.get("name", ""),
                "purpose": detail.get("purpose", ""),
                "for_beginners": detail.get("use", "")
            })

    similar = find_similar_guidelines(query, all_guidelines)
    if similar:
        response = ""
        for g, score in similar[:3]:
            response += f"📘 **{g['code']} - {g['title']}**\nPurpose: {g['purpose']}\nUse: {g['for_beginners']}\n\n"
        return response

    return "❌ No matching or similar guideline found."
