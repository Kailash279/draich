# RL_GEN_AI.py

import os
import re
import json
import datetime
import wikipedia
import logging
import hashlib
import sqlite3

# Setup logging
logging.basicConfig(level=logging.INFO, filename="app.log")
logger = logging.getLogger(__name__)

# Memory paths
MEMORY_DB = os.path.join(os.path.dirname(__file__), "memory.db")

# Ensure SQLite DB exists
def init_sql_memory():
    try:
        conn = sqlite3.connect(MEMORY_DB)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                question TEXT,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB init failed: {e}")
        raise

init_sql_memory()

# Utilities
def clean_text(text):
    return re.sub(r'\W+', ' ', text.lower()).strip() if isinstance(text, str) else ""

def time_based_greeting():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good Morning!"
    elif hour < 17:
        return "Good Afternoon!"
    else:
        return "Good Evening!"

# SQL Memory Functions
def search_memory(query):
    cleaned_query = clean_text(query)
    if not cleaned_query:
        return None

    try:
        conn = sqlite3.connect(MEMORY_DB)
        c = conn.cursor()
        c.execute("SELECT response FROM memory WHERE question = ?", (cleaned_query,))
        row = c.fetchone()
        conn.close()
        if row:
            logger.info(f"Memory match found: {cleaned_query}")
            return row[0]
    except Exception as e:
        logger.error(f"Memory search error: {e}")
    return None

def save_to_memory(query, response):
    cleaned_query = clean_text(query)
    if not cleaned_query or not response:
        return

    try:
        conn = sqlite3.connect(MEMORY_DB)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO memory (id, question, response)
            VALUES (?, ?, ?)
        """, (hashlib.md5(cleaned_query.encode()).hexdigest(), cleaned_query, response.strip()))
        conn.commit()
        conn.close()
        logger.info(f"Saved to memory: {cleaned_query}")
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")

# Wikipedia fallback
def fetch_wikipedia_answer(query):
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
        return "Unable to fetch from Wikipedia."

# DQN placeholder (mocked decision strategy)
def dqn_decision(query):
    """
    Placeholder for DQN logic. You can later plug a real model here.
    """
    keywords = ["ich", "guideline", "stability", "efficacy"]
    if any(k in query.lower() for k in keywords):
        return "use_local_guidelines"
    return "ask_wikipedia"

# RL Dynamic Response Logic
def generate_dynamic_response(query):
    if not query or len(query.strip()) < 3:
        return "Please provide a more specific question."

    # 1. Check memory
    mem = search_memory(query)
    if mem:
        return mem

    # 2. DQN decides next best move
    action = dqn_decision(query)

    if action == "use_local_guidelines":
        return "Please refer to the structured ICH guidelines database for this topic."

    # 3. Wikipedia fallback
    wiki = fetch_wikipedia_answer(query)
    if wiki:
        save_to_memory(query, wiki)
        return wiki

    return "Sorry, I couldn't find any relevant information."

# sql_memory.py
import mysql.connector
import hashlib
import datetime

# 🔐 MySQL connection config
config = {
    "host": "localhost",         # or your host IP
    "user": "root",              # your MySQL username
    "password": "your_password", # your MySQL password
    "database": "ich_chatbot"    # your DB name
}

def clean_text(text):
    return ''.join(e for e in text.lower() if e.isalnum() or e.isspace()).strip()

def save_to_memory(question, response):
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    
    question_clean = clean_text(question)
    response_clean = response.strip()
    qid = hashlib.md5(question_clean.encode()).hexdigest()

    sql = """
    INSERT INTO memory (id, question, response)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE response = VALUES(response);
    """
    cursor.execute(sql, (qid, question_clean, response_clean))
    conn.commit()
    cursor.close()
    conn.close()

def search_memory(question):
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    
    question_clean = clean_text(question)
    qid = hashlib.md5(question_clean.encode()).hexdigest()

    cursor.execute("SELECT response FROM memory WHERE id = %s", (qid,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    return result[0] if result else None
