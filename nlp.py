def classify_query(text):
    text = text.lower()

    if any(word in text for word in ["hello", "hi", "namaste"]):
        return "Greeting"
    elif any(word in text for word in ["bye", "goodbye", "alvida"]):
        return "Farewell"
    elif any(word in text for word in ["madad", "help", "support", "query"]):
        return "Support Query"
    elif any(word in text for word in ["formulation", "banane", "development", "pdr"]):
        return "Product Development"
    elif any(word in text for word in ["stability", "shelf life", "q1"]):
        return "Stability Testing"
    elif any(word in text for word in ["validation", "analytical", "test", "method"]):
        return "Analytical Validation"
    else:
        return "General Query"
