import nltk
import spacy
import subprocess
import sys

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    print("NLTK data downloaded successfully!")

def download_spacy_model():
    """Download spaCy English model"""
    print("Downloading spaCy English model...")
    try:
        spacy.load('en_core_web_sm')
        print("spaCy model already exists!")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("spaCy model downloaded successfully!")

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Requirements installed successfully!")

if __name__ == "__main__":
    print("Starting setup...")
    install_requirements()
    download_nltk_data()
    download_spacy_model()
    print("Setup completed successfully!") 