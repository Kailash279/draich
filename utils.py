# utils.py

from PyPDF2 import PdfReader
from transformers import pipeline

# Load the summarization pipeline globally
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ========== Extract Text from PDF ==========
def extract_text_from_pdf(file_path):
    """
    Extract text from a given PDF file path.
    """
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

# ========== Summarize Text ==========
def summarize_text(text, max_length=150, min_length=30):
    """
    Summarize long text using transformer-based summarization.
    """
    try:
        if not text.strip():
            return "No text to summarize."

        # Limit input to avoid max token errors
        short_text = text[:1024]
        summary = summarizer(short_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing text: {e}"
