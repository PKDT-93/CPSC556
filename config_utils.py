# config_utils.py
"""
Stores configuration variables and utility functions for the malware detection pipeline.
Loads sensitive variables from a .env file.
Includes lists for NLP context feature generation.
"""
import os
import re
import pandas as pd
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()
print("Attempted to load environment variables from .env file.")

# --- Configuration ---

# Kaggle Settings
KAGGLE_DATASET_ID = "ramoliyafenil/text-based-cyber-threat-detection"
DOWNLOAD_DIR = "./kaggle_data"
MAIN_CSV_FILE = "cyber-threat-intelligence_all.csv"
LOCAL_CSV_PATH = os.path.join(DOWNLOAD_DIR, MAIN_CSV_FILE)

# DeepSeek API Settings
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
API_CALL_DELAY_SECONDS = 1.5

# Dictionary & Model Settings
MIN_TERM_LENGTH = 4
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42
DICTIONARY_CACHE_FILE = "malware_dictionary_cache.txt"

# --- Feature Engineering Settings ---
# Network features are confirmed absent/unused based on data analysis
NETWORK_NUMERICAL_FEATURES = []
NETWORK_CATEGORICAL_FEATURES = []

# NLP Context Features Configuration
# Keywords suggesting informational context (lowercase)
INFO_KEYWORDS = [
    'report', 'analysis', 'threat intelligence', 'vulnerability', 'advisory',
    'bulletin', 'detected', 'detection', 'signature', 'research', 'blog post',
    'news', 'article', 'mitigation', 'indicator', 'ioc', 'campaign analysis',
    'security brief', 'overview', 'summary', 'presentation', 'webinar'
]
# NER Labels (from spaCy) suggesting informational context (e.g., organizations, products)
INFO_NER_LABELS = [
    'ORG',       # Companies, agencies, institutions.
    'PRODUCT',   # Objects, vehicles, foods, etc. (can include software products)
    'GPE',       # Geopolitical entities (countries, cities, states) - might indicate report origin
    'PERSON'     # People's names - might indicate researchers/authors
]
# Feature names for the new context columns
INFO_KEYWORD_FEATURE = 'has_info_keywords'
INFO_ENTITY_FEATURE = 'has_info_entities'

# Original dictionary flag feature and target
TEXT_MALWARE_FLAG_FEATURE = 'malware_flag'
TARGET_FEATURE = 'true_label'

# --- Utility Functions ---
def normalize_name(name: str) -> str:
    """
    Normalizes a potential malware name by lowercasing, replacing separators
    with spaces, and stripping whitespace.
    """
    if not isinstance(name, str): name = str(name) # Ensure it's a string
    name = name.lower()
    name = re.sub(r'[\.\-_]', ' ', name) # Replace '.', '-', '_' with space
    name = re.sub(r'\s+', ' ', name).strip() # Collapse multiple spaces and strip ends
    return name

def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing: lowercase, remove non-alphanumeric/space chars,
    collapse whitespace.
    """
    if not isinstance(text, str): text = str(text) # Ensure it's a string
    text = text.lower()
    # Remove characters that are not word characters (alphanumeric + underscore) or whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip() # Remove leading/trailing whitespace

def map_binary_label(label) -> int:
    """
    Maps the input label (expected to be 'malware' or something else)
    to a binary integer (1 for malware, 0 otherwise).
    """
    # Check if label is a string and equals 'malware' case-insensitively
    return 1 if isinstance(label, str) and label.lower() == 'malware' else 0

# --- Helper for checking API key ---
def check_api_key():
    """Checks if the DeepSeek API key was loaded from the environment."""
    if not DEEPSEEK_API_KEY:
        print("WARNING: DEEPSEEK_API_KEY not found in environment variables or .env file.")
        print("Dictionary expansion will be limited or disabled if cache doesn't exist.")
        return False
    return True

print("Config and Utils loaded.")
# Check the API key status when the module is imported
check_api_key()