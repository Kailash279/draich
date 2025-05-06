import os
import datetime
import wikipedia
import json
import re

# Directory to store memory
MEMORY_DIR = "rl_memory"
if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

def time_based_greeting():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good Morning!"
    elif hour < 17:
        return "Good Afternoon!"
    else:
        return "Good Evening!"

def clean_text(text):
    return re.sub(r'\W+', ' ', text.lower()).strip()

def search_memory(user_input):
    cleaned_input = clean_text(user_input)
    for filename in os.listdir(MEMORY_DIR):
        with open(os.path.join(MEMORY_DIR, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            if cleaned_input in data.get("question", ""):
                return data.get("response")
    return None

def learn_response(user_input, bot_response):
    cleaned_input = clean_text(user_input)
    file_name = f"{cleaned_input[:30].replace(' ', '_')}.json"
    memory_data = {
        "question": cleaned_input,
        "response": bot_response
    }
    with open(os.path.join(MEMORY_DIR, file_name), "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=4)

def fetch_wikipedia_answer(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception as e:
        return "Sorry, I couldn't find anything relevant on Wikipedia."

def reinforcement_chatbot():
    print("🤖 Cosmos RL Chatbot Activated!")
    print(time_based_greeting())
    print("Type 'exit' to end the chat.\n")

    while True:
        user_input = input("You: ")

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

        feedback = input("Do you want to save this response for future? (yes/no): ")
        if feedback.strip().lower() == "yes":
            learn_response(user_input, wiki_reply)
            print("✅ Learned and saved to memory.")

if __name__ == "__main__":
    reinforcement_chatbot()
