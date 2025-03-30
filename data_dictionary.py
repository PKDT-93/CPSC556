# data_dictionary.py
"""
Handles Kaggle data downloading/loading and builds the malware dictionary.
**Updated for Cyber-Threat-Intelligence-Custom-Data_new_processed.csv**
Includes check for 'relations' and 'id_X' columns needed for graph features.
"""
import os
import subprocess
import zipfile
import pandas as pd
import time
import ast
import re
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from tqdm import tqdm

import config_utils as cfg

# --- Data Loading Functions ---
def download_kaggle_dataset():
    # Function remains the same as previous version for processed data
    print(f"\n--- Checking/Downloading Kaggle dataset: {cfg.KAGGLE_DATASET_ID} ---")
    if not os.path.exists(cfg.DOWNLOAD_DIR):
        os.makedirs(cfg.DOWNLOAD_DIR)
        print(f"Created download directory: {cfg.DOWNLOAD_DIR}")
    if os.path.exists(cfg.LOCAL_CSV_PATH):
        print(f"Dataset file already found locally at: {cfg.LOCAL_CSV_PATH}")
        print("Skipping download.")
        return True
    try:
        result = subprocess.run(['kaggle', '--version'], check=True, capture_output=True, text=True)
        print(f"Kaggle CLI found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: Kaggle API command-line tool not found or not configured.")
        print("Please install and configure it: pip install kaggle; kaggle setup")
        print("Alternatively, place the dataset manually at:", cfg.LOCAL_CSV_PATH)
        return False
    except Exception as e:
         print(f"\nAn unexpected error occurred checking Kaggle CLI: {e}")
         return False

    command = ['kaggle', 'datasets', 'download', '-d', cfg.KAGGLE_DATASET_ID, '-p', cfg.DOWNLOAD_DIR, '--unzip']
    print(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Kaggle download output:\n{result.stdout}")
        if result.stderr:
             print(f"Kaggle download error output:\n{result.stderr}")
        if os.path.exists(cfg.LOCAL_CSV_PATH):
            print(f"Dataset downloaded and unzipped successfully. Found: {cfg.LOCAL_CSV_PATH}")
            return True
        else:
            print(f"ERROR: Kaggle command ran, but expected file not found at {cfg.LOCAL_CSV_PATH}")
            files_in_dir = os.listdir(cfg.DOWNLOAD_DIR)
            print(f"Files found in download directory: {files_in_dir}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\nERROR downloading dataset using Kaggle CLI: {e}")
        print(f"Command output (stdout):\n{e.stdout}")
        print(f"Command error output (stderr):\n{e.stderr}")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred during Kaggle download: {e}")
        return False


def load_main_dataframe() -> pd.DataFrame | None:
    print(f"\n--- Loading Main Dataset from: {cfg.LOCAL_CSV_PATH} ---")
    try:
        df = pd.read_csv(cfg.LOCAL_CSV_PATH)
        print(f"Main dataset loaded successfully. Shape: {df.shape}")

        # *** UPDATED COLUMN CHECK ***
        # Check for essential columns for text, entities, and graph features
        required_cols = ['text', 'relations', 'id_1', 'label_1', 'start_offset_1', 'end_offset_1']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             print(f"ERROR: Essential columns missing (checked for: {', '.join(required_cols)}).")
             print("Ensure the CSV has 'text', 'relations', and the 'id_X', 'label_X', 'start_offset_X', 'end_offset_X' pattern.")
             return None
        print("Found required columns for text, relations, and entity pattern.")

        if df.empty:
            print("ERROR: Loaded dataframe is empty.")
            return None
        return df
    except FileNotFoundError:
        print(f"ERROR: Local CSV file not found at {cfg.LOCAL_CSV_PATH}.")
        return None
    except pd.errors.EmptyDataError:
        print(f"ERROR: The CSV file at {cfg.LOCAL_CSV_PATH} is empty.")
        return None
    except Exception as e:
        print(f"ERROR loading main dataset CSV: {e}")
        return None

# --- Dictionary Building Functions ---
# (No changes needed in client init, _extract_initial_malware_names,
# _query_deepseek_for_aliases, or build_malware_dictionary - they focus
# only on text/label_X/offset_X for the dictionary itself)

# Initialize DeepSeek client
client = None
if cfg.check_api_key():
    try:
        client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
        print("DeepSeek client initialized.")
    except Exception as e:
        print(f"ERROR: Failed to initialize DeepSeek client: {e}")
        client = None
else:
    print("DeepSeek client not initialized due to missing API key.")

def is_deepseek_client_active() -> bool:
    return client is not None

def _extract_initial_malware_names(df: pd.DataFrame) -> set:
    # Function remains the same as previous version for processed data
    initial_malware_names = set()
    processed_rows = 0
    malware_mention_count = 0
    error_count = 0
    if 'text' not in df.columns: return initial_malware_names
    max_index = 0
    for col in df.columns:
        match = re.match(r'label_(\d+)', col)
        if match: max_index = max(max_index, int(match.group(1)))
    if max_index == 0: return initial_malware_names
    print(f"Found entity columns up to index: {max_index}")
    df_filled_text = df['text'].fillna('').astype(str)
    for index, text_str in tqdm(df_filled_text.items(), total=len(df), desc="Scanning entities"):
        processed_rows += 1
        text_len = len(text_str)
        for i in range(1, max_index + 1):
            label_col, start_col, end_col = f'label_{i}', f'start_offset_{i}', f'end_offset_{i}'
            if not all(c in df.columns for c in [label_col, start_col, end_col]): continue
            label = df.loc[index, label_col]
            start_offset = df.loc[index, start_col]
            end_offset = df.loc[index, end_col]
            if isinstance(label, str) and label.lower() == 'malware':
                malware_mention_count += 1
                try:
                    start = int(pd.to_numeric(start_offset, errors='coerce'))
                    end = int(pd.to_numeric(end_offset, errors='coerce'))
                    if pd.notna(start) and pd.notna(end) and 0 <= start < end <= text_len:
                        malware_name = text_str[start:end].strip()
                        if malware_name: initial_malware_names.add(malware_name)
                    else: error_count += 1
                except (ValueError, TypeError): error_count += 1
                except Exception as e: print(f"Unexpected error row {index}, Entity {i}: {e}"); error_count += 1
    print(f"Processed {processed_rows} rows. Found {malware_mention_count} 'malware' mentions.")
    if error_count > 0: print(f"Warning: Encountered {error_count} errors/invalid offsets during entity extraction.")
    print(f"Extracted {len(initial_malware_names)} unique non-empty potential malware names.")
    return initial_malware_names

def _query_deepseek_for_aliases(malware_name: str) -> list | None:
    # Function remains the same as previous version for processed data
    if not client or not isinstance(malware_name, str) or not malware_name.strip(): return []
    prompt = (f"List only known aliases or alternate names for the malware \"{malware_name}\". Return only a comma-separated list (e.g., alias1,alias2,alias three). If you don't know any aliases or the input is not a malware name, return an empty string. Do not explain anything or add introductory text. Do not use markdown.")
    try:
        response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=100, stream=False, timeout=20)
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content: return []
        content = response.choices[0].message.content.strip()
        if not content or any(p in content.lower() for p in ["i couldn't find", "i do not have information", "no known aliases", "not a malware", "here are some aliases:"]) and ',' not in content: return []
        aliases = [cfg.normalize_name(a) for a in content.split(',') if a.strip()]
        cleaned = [a for a in aliases if len(a) >= cfg.MIN_TERM_LENGTH and not a.startswith("i don't")]
        return cleaned
    except RateLimitError: print(f"  -> Rate limit hit for '{malware_name}'. Waiting..."); time.sleep(cfg.API_CALL_DELAY_SECONDS * 5); return None
    except APITimeoutError: print(f"  -> DeepSeek request timed out for '{malware_name}'. Skipping."); return []
    except APIError as e: print(f"  -> DeepSeek API Error for '{malware_name}': {e}. Skipping."); return []
    except Exception as e: print(f"  -> Unexpected Error during DeepSeek query for '{malware_name}': {type(e).__name__} - {e}. Skipping."); return []

def build_malware_dictionary(df: pd.DataFrame) -> tuple[set, bool]:
    # Function remains the same as previous version for processed data
    print("\n--- Building Malware Dictionary (Processed Data Version) ---")
    cache_loaded = False; final_dictionary = set()
    if os.path.exists(cfg.DICTIONARY_CACHE_FILE):
        print(f"Loading malware dictionary from cache file: {cfg.DICTIONARY_CACHE_FILE}")
        try:
            with open(cfg.DICTIONARY_CACHE_FILE, 'r', encoding='utf-8') as f: cached_dict = {line.strip() for line in f if line.strip()}
            print(f"Successfully loaded {len(cached_dict)} terms from cache.")
            if not cached_dict: print("Warning: Cache file was empty. Proceeding to rebuild dictionary.")
            else: final_dictionary = cached_dict; cache_loaded = True; return final_dictionary, cache_loaded
        except Exception as e: print(f"Error loading dictionary from cache: {e}. Proceeding to rebuild.")
    else: print(f"Dictionary cache file not found ({cfg.DICTIONARY_CACHE_FILE}). Building dictionary from scratch...")
    print("Extracting initial malware names from dataset (flattened structure)...")
    initial_names = _extract_initial_malware_names(df)
    if not initial_names: print("Warning: No initial malware names extracted. Dictionary will be empty."); return set(), cache_loaded
    expanded_dict = {cfg.normalize_name(name) for name in initial_names}
    expanded_dict = {name for name in expanded_dict if len(name) >= cfg.MIN_TERM_LENGTH}
    print(f"Added {len(expanded_dict)} initial normalized names (>= {cfg.MIN_TERM_LENGTH} chars).")
    if client:
        print(f"Querying DeepSeek for aliases for {len(initial_names)} unique initial names...")
        api_failures = 0; successful_queries = 0; aliases_found_count = 0
        names_to_query = sorted([name for name in initial_names if name and len(name) >= cfg.MIN_TERM_LENGTH])
        for name in tqdm(names_to_query, desc="Querying DeepSeek"):
            aliases = _query_deepseek_for_aliases(name)
            if aliases is None: api_failures += 1; continue
            successful_queries += 1
            if aliases: original_size = len(expanded_dict); expanded_dict.update(aliases); aliases_found_count += (len(expanded_dict) - original_size)
            time.sleep(cfg.API_CALL_DELAY_SECONDS)
        print(f"DeepSeek querying complete. Successful queries: {successful_queries}/{len(names_to_query)}, API failures: {api_failures}, New aliases: {aliases_found_count}")
    else: print("DeepSeek client not available or not initialized. Skipping alias expansion.")
    final_dictionary = expanded_dict; final_size = len(final_dictionary)
    print(f"🧠 Final dictionary size: {final_size}")
    if final_dictionary and not cache_loaded:
        print(f"Saving dictionary ({final_size} terms) to cache file: {cfg.DICTIONARY_CACHE_FILE}")
        try:
            sorted_terms = sorted(list(final_dictionary));
            with open(cfg.DICTIONARY_CACHE_FILE, 'w', encoding='utf-8') as f:
                for term in sorted_terms: f.write(term + '\n')
            print("Dictionary saved successfully.")
        except Exception as e: print(f"Error saving dictionary to cache: {e}")
    elif not final_dictionary: print("Warning: Final dictionary is empty. Not saving cache file.")
    return final_dictionary, cache_loaded