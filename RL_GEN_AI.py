import os
import datetime
import wikipedia
import json
import re
import logging
import hashlib

# ========== Logger Setup ==========
logging.basicConfig(level=logging.INFO, filename="app.log")
logger = logging.getLogger(__name__)

# ========== Memory Directory ==========
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "rl_memory")

def ensure_memory_dir():
    """Ensure the memory directory exists and is writable."""
    try:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        test_file = os.path.join(MEMORY_DIR, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        logger.error(f"Memory directory error: {e}")
        raise RuntimeError(f"Cannot access memory directory: {e}")

ensure_memory_dir()

# ========== Utilities ==========

def time_based_greeting():
    """Return a time-based greeting."""
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good Morning!"
    elif hour < 17:
        return "Good Afternoon!"
    else:
        return "Good Evening!"

def clean_text(text):
    """Normalize text to lowercase and remove non-word characters."""
    return re.sub(r'\W+', ' ', text.lower()).strip() if isinstance(text, str) else ""

# ========== Memory Functions ==========

def search_memory(user_input):
    """Return saved response if user_input exists in memory."""
    cleaned_input = clean_text(user_input)
    if not cleaned_input:
        return None

    try:
        for filename in os.listdir(MEMORY_DIR):
            if filename.endswith(".json"):
                with open(os.path.join(MEMORY_DIR, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("question") == cleaned_input:
                        logger.info(f"Memory match found: {cleaned_input}")
                        return data.get("response")
    except Exception as e:
        logger.error(f"Memory search error: {e}")
    return None

def save_to_memory(user_input, bot_response):
    """Save cleaned user_input and bot_response to JSON memory."""
    cleaned_input = clean_text(user_input)
    if not cleaned_input or not bot_response:
        return

    file_hash = hashlib.md5(cleaned_input.encode()).hexdigest()
    file_path = os.path.join(MEMORY_DIR, f"{file_hash}.json")

    memory_data = {
        "question": cleaned_input,
        "response": bot_response.strip()
    }

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=4)
        logger.info(f"Saved memory: {file_path}")
    except Exception as e:
        logger.error(f"Error saving memory: {e}")

# ========== Wikipedia Fallback ==========

def fetch_wikipedia_answer(query):
    """Return Wikipedia summary for a query."""
    cleaned_query = clean_text(query)
    if not cleaned_query:
        return "Sorry, I couldn't process your query."

    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(cleaned_query, sentences=2, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {', '.join(e.options[:3])}."
    except wikipedia.exceptions.PageError:
        return "No relevant page found on Wikipedia."
    except Exception as e:
        logger.error(f"Wikipedia error: {e}")
        return "Unable to fetch information from Wikipedia right now."

# ========== RL Engine ==========

def generate_dynamic_response(query):
    """Check memory first, else fetch from Wikipedia and store"""
