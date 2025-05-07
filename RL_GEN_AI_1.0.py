import os
import datetime
import wikipedia
import json
import re
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, filename="app.log")
logger = logging.getLogger(__name__)

# Directory to store memory
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "rl_memory")

def ensure_memory_dir():
    """Ensure the memory directory exists and is writable."""
    try:
        if not os.path.exists(MEMORY_DIR):
            os.makedirs(MEMORY_DIR)
            logger.info(f"Created memory directory: {MEMORY_DIR}")
        # Test write permissions
        test_file = os.path.join(MEMORY_DIR, ".test_write")
        with open(test_file, "w") as f:
            f.write("")
        os.remove(test_file)
    except Exception as e:
        logger.error(f"Failed to create or access memory directory {MEMORY_DIR}: {e}")
        raise RuntimeError(f"Cannot access memory directory: {e}")

# Initialize memory directory
ensure_memory_dir()

def time_based_greeting():
    """Return a time-based greeting based on the current hour."""
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good Morning!"
    elif hour < 17:
        return "Good Afternoon!"
    else:
        return "Good Evening!"

def clean_text(text):
    """Clean and normalize text by removing non-word characters and converting to lowercase."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\W+', ' ', text.lower()).strip()

def search_memory(user_input):
    """Search memory files for a matching question and return its response."""
    cleaned_input = clean_text(user_input)
    if not cleaned_input:
        return None

    try:
        for filename in os.listdir(MEMORY_DIR):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(MEMORY_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("question") == cleaned_input:  # Exact match
                        logger.info(f"Found memory match for question: {cleaned_input}")
                        return data.get("response")
            except Exception as e:
                logger.error(f"Error reading memory file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error accessing memory directory: {e}")
        return None

def learn_response(user_input, bot_response):
    """Save a question-response pair to memory with a unique file name."""
    cleaned_input = clean_text(user_input)
    if not cleaned_input or not isinstance(bot_response, str):
        logger.error("Invalid input or response for learning")
        return

    # Use hash of cleaned input for unique, safe file names
    hash_input = hashlib.md5(cleaned_input.encode()).hexdigest()
    file_name = f"{hash_input}.json"
    file_path = os.path.join(MEMORY_DIR, file_name)

    memory_data = {
        "question": cleaned_input,
        "response": bot_response.strip()
    }

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=4)
        logger.info(f"Saved response to memory: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save memory file {file_path}: {e}")

def fetch_wikipedia_answer(query):
    """Fetch a summary from Wikipedia for the given query."""
    cleaned_query = clean_text(query)
    if not cleaned_query:
        return "Sorry, I couldn't process the query."

    try:
        wikipedia.set_lang("en")  # Ensure English Wikipedia
        summary = wikipedia.summary(cleaned_query, sentences=2, auto_suggest=True)
        logger.info(f"Wikipedia summary fetched for query: {cleaned_query}")
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        logger.warning(f"Wikipedia disambiguation error for query: {cleaned_query}")
        return f"Multiple options found: {', '.join(e.options[:3])}. Please be more specific."
    except wikipedia.exceptions.PageError:
        logger.warning(f"Wikipedia page not found for query: {cleaned_query}")
        return "Sorry, I couldn't find anything relevant on Wikipedia."
    except Exception as e:
        logger.error(f"Wikipedia fetch error: {e}")
        return "Sorry, I couldn't connect to Wikipedia."

def reinforcement_chatbot():
    """Run the Cosmos RL Chatbot with memory and Wikipedia integration."""
    print("🤖 Cosmos RL Chatbot Activated!")
    print(time_based_greeting())
    print("Type 'exit' to end the chat.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Cosmos: Please enter a valid question.")
            continue

        if user_input.lower() == "exit":
            print("Cosmos: Goodbye, take care! 😊")
            break

        memory_reply = search_memory(user_input)
        if memory_reply:
            print(f"Cosmos (from memory): {memory_reply}")
            continue

        print("Cosmos: I don't know that yet. Let me try searching Wikipedia...")
        wiki_reply = fetch_wikipedia_answer(user_input)
        print(f"Cosmos (Wikipedia): {wiki_reply}")

        while True:
            feedback = input("Do you want to save this response for future? (yes/no): ").strip().lower()
            if feedback in ["yes", "no"]:
                break
            print("Cosmos: Please enter 'yes' or 'no'.")

        if feedback == "yes":
            learn_response(user_input, wiki_reply)
            print("✅ Learned and saved to memory.")

if __name__ == "__main__":
    reinforcement_chatbot()