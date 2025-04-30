import streamlit as st
import json

def load_guideline_data():
    with open("ich_guidelines_full_combined.json", "r", encoding="utf-8") as f:
        return json.load(f)

def search_guidelines(query):
    data = load_guideline_data()
    query_lower = query.lower()
    for item in data:
        if query_lower in item["title"].lower():
            return f"**{item['code']}** – {item['title']}\n\n{item['purpose']}"
    return "No matching guideline found."

st.title("Test ICH Guideline Search")

prompt = st.text_input("Ask ICH related question:")

if prompt:
    st.write("Searching...")
    result = search_guidelines(prompt)
    st.write(result)
