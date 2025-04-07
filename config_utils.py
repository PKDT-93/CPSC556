# config_utils.py
"""
Stores configuration variables and utility functions for the malware detection pipeline.
Loads sensitive variables from a .env file.
Includes lists for NLP context feature generation.
**Added configuration for Graph Features**
"""
import os
import re
import pandas as pd
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()
print("Attempted to load environment variables from .env file.")

# --- Configuration ---

# Dataset Settings (Using Processed Dataset)
KAGGLE_DATASET_ID = "ramoliyafenil/text-based-cyber-threat-detection" # Placeholder
DOWNLOAD_DIR = "./kaggle_data"
MAIN_CSV_FILE = "Cyber-Threat-Intelligence-Custom-Data_new_processed.csv"
LOCAL_CSV_PATH = os.path.join(DOWNLOAD_DIR, MAIN_CSV_FILE)

# DeepSeek API Settings
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
API_CALL_DELAY_SECONDS = 1.5

# Dictionary & Model Settings
MIN_TERM_LENGTH = 4
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42
DICTIONARY_CACHE_FILE = "malware_dictionary_cache_processed_data.txt"

# --- Feature Engineering Settings ---

# NLP Context Features Configuration
INFO_KEYWORDS = [
    'report', 'analysis', 'threat intelligence', 'vulnerability', 'advisory',
    'bulletin', 'detected', 'detection', 'signature', 'research', 'blog post',
    'news', 'article', 'mitigation', 'indicator', 'ioc', 'campaign analysis',
    'security brief', 'overview', 'summary', 'presentation', 'webinar', "biopass", "rat", "malware", "backdoor", "exe", "figure", "loader", "used", "win64", "version",
    "file", "code", "ransomware", "attacks", "malicious", "attack", "samples", "family", "using",
    "command", "group", "variant", "use", "plugx", "protux", "anel", "new", "server", "cyclops", "infection"
]
INFO_NER_LABELS = [
    'ORG', 'PRODUCT', 'GPE', 'PERSON'
]
# Feature names for NLP context columns
INFO_KEYWORD_FEATURE = 'has_info_keywords'
INFO_ENTITY_FEATURE = 'has_info_entities' # Still defined, but might be unused if spaCy fails

# Dictionary flag feature
TEXT_MALWARE_FLAG_FEATURE = 'malware_flag'

# *** NEW: Graph Feature Names ***
# Simple example features: Max degree of a malware node in the row's graph, total edges
MAX_MALWARE_DEGREE_FEATURE = 'graph_max_malware_degree'
TOTAL_EDGES_FEATURE = 'graph_total_edges'
# Add more features here if desired (e.g., total_nodes, avg_malware_degree)

# Target feature name
TARGET_FEATURE = 'true_label'

# --- Utility Functions (Remain the same) ---
def normalize_name(name: str) -> str:
    if not isinstance(name, str): name = str(name)
    name = name.lower()
    name = re.sub(r'[\.\-_]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def preprocess_text(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def map_binary_label(label) -> int:
    return 1 if isinstance(label, str) and label.lower() == 'malware' else 0

# --- Helper for checking API key (Remains the same) ---
def check_api_key():
    if not DEEPSEEK_API_KEY:
        print("WARNING: DEEPSEEK_API_KEY not found. Dictionary expansion may be limited.")
        return False
    print("DeepSeek API Key found.")
    return True

print("Config and Utils loaded (Added Graph Feature Config).")
check_api_key()
