# utils.py
from PyPDF2 import PdfReader
from transformers import pipeline

# Load the summarization pipeline once globally
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"

def summarize_text(text, max_length=150, min_length=30):
    try:
        if not text.strip():
            return "No text to summarize."
        short_text = text[:1024]  # to avoid long input errors
        summary = summarizer(short_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing text: {e}"
