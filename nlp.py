import re
from textblob import TextBlob

def preprocess_text(text):
    """
    यह function text को clean और normalize करता है।
    """
    text = text.lower()  # Lowercase conversion
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def classify_query(text):
    """
    यह function query को classify करता है और sentiment analysis करता है।
    """
    processed_text = preprocess_text(text)
    
    # Sentiment analysis using TextBlob
    blob = TextBlob(processed_text)
    sentiment = blob.sentiment.polarity
    
    # Query Classification Logic (Basic Example)
    if "hello" in processed_text or "hi" in processed_text:
        return "Greeting"
    elif "help" in processed_text or "support" in processed_text:
        return "Support Query"
    elif "bye" in processed_text or "goodbye" in processed_text:
        return "Farewell"
    elif sentiment > 0:
        return "Positive Statement"
    elif sentiment < 0:
        return "Negative Statement"
    else:
        return "General Query"

if __name__ == "__main__":
    # Test the function
    user_input = input("Enter your query: ")
    result = classify_query(user_input)
    print(f"Query Type: {result}")
