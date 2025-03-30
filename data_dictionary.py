# data_dictionary.py
"""
Handles Kaggle data downloading/loading and builds the malware dictionary using DeepSeek,
with caching to avoid repeated API calls.
"""
import os
import subprocess
import zipfile
import pandas as pd
import time
import ast
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from tqdm import tqdm

# Import from our config/utils module
import config_utils as cfg

# --- Data Loading Functions ---
def download_kaggle_dataset():
    print(f"\n--- Checking/Downloading Kaggle dataset: {cfg.KAGGLE_DATASET_ID} ---")
    if not os.path.exists(cfg.DOWNLOAD_DIR):
        os.makedirs(cfg.DOWNLOAD_DIR)
        print(f"Created download directory: {cfg.DOWNLOAD_DIR}")
    if os.path.exists(cfg.LOCAL_CSV_PATH):
        print(f"Dataset file already found locally at: {cfg.LOCAL_CSV_PATH}")
        print("Skipping download.")
        return True
    try:
        # Check if kaggle command exists and works
        result = subprocess.run(['kaggle', '--version'], check=True, capture_output=True, text=True)
        print(f"Kaggle CLI found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: Kaggle API command-line tool not found or not configured.")
        print("Please install and configure it: pip install kaggle; kaggle setup")
        return False
    except Exception as e:
         print(f"\nAn unexpected error occurred checking Kaggle CLI: {e}")
         return False

    command = ['kaggle', 'datasets', 'download', '-d', cfg.KAGGLE_DATASET_ID, '-p', cfg.DOWNLOAD_DIR, '--unzip']
    print(f"Executing command: {' '.join(command)}")
    try:
        # Run the download command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Kaggle download output:\n{result.stdout}")
        if result.stderr:
             print(f"Kaggle download error output:\n{result.stderr}")
        # Check if the specific expected file exists after download/unzip
        if os.path.exists(cfg.LOCAL_CSV_PATH):
            print(f"Dataset downloaded and unzipped successfully. Found: {cfg.LOCAL_CSV_PATH}")
            return True
        else:
            print(f"ERROR: Kaggle command ran, but expected file not found at {cfg.LOCAL_CSV_PATH}")
            print(f"Please check the contents of {cfg.DOWNLOAD_DIR}")
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
        # More robust check for required columns
        required_cols = ['text', 'label', 'entities'] # Add 'entities' as it's needed later
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             print(f"ERROR: Required columns missing from CSV: {', '.join(missing_cols)}")
             return None
        # Optional: Check for empty dataframe
        if df.empty:
            print("ERROR: Loaded dataframe is empty.")
            return None
        return df
    except FileNotFoundError:
        print(f"ERROR: Local CSV file not found at {cfg.LOCAL_CSV_PATH}.")
        print("Attempt running the script again to trigger download, or check the path.")
        return None
    except pd.errors.EmptyDataError:
        print(f"ERROR: The CSV file at {cfg.LOCAL_CSV_PATH} is empty.")
        return None
    except Exception as e:
        print(f"ERROR loading main dataset CSV: {e}")
        return None


# --- Dictionary Building Functions ---

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
    """Checks if the module-level DeepSeek client was successfully initialized."""
    return client is not None

def _extract_initial_malware_names(df: pd.DataFrame) -> set:
    # ... (Keep this internal function exactly as before) ...
    initial_malware_names = set()
    malformed_entity_count = 0
    processed_rows = 0
    if 'entities' in df.columns and 'text' in df.columns:
         # Fill NA in 'entities' and 'text' to avoid errors during iteration
        df_filled = df[['entities', 'text']].fillna({'entities': '[]', 'text': ''})

        for _, row in tqdm(df_filled.iterrows(), total=df_filled.shape[0], desc="Scanning entities"):
            processed_rows += 1
            entities_str = row['entities']
            text_str = row['text']

            if not isinstance(entities_str, str) or not entities_str.strip():
                continue # Skip if entities is not a string or empty

            try:
                # Safely evaluate the string representation of the list
                entities_list = ast.literal_eval(entities_str)
                if not isinstance(entities_list, list):
                     malformed_entity_count += 1
                     continue # Skip if it's not a list after eval

                for ent in entities_list:
                    # Check if entity is a dict and has the required keys and label
                    if isinstance(ent, dict) and \
                       ent.get('label', '').lower() == 'malware' and \
                       'start_offset' in ent and 'end_offset' in ent:

                        start, end = ent['start_offset'], ent['end_offset']

                        # Validate offsets and text type
                        if isinstance(start, int) and isinstance(end, int) and \
                           isinstance(text_str, str) and 0 <= start < end <= len(text_str):
                            malware_name = text_str[start:end]
                            initial_malware_names.add(malware_name.strip())
                        # Optional: Add logging here for invalid offset errors if needed
                        # else:
                        #     print(f"Warning: Invalid offsets {start}-{end} for text length {len(text_str)}")

            except (ValueError, SyntaxError, TypeError) as e:
                # Catch errors during literal_eval or processing
                # print(f"Debug: Error processing entities '{entities_str[:100]}...': {e}") # Uncomment for debug
                malformed_entity_count += 1
                continue
            except Exception as e:
                 # Catch any other unexpected errors
                 print(f"Unexpected error processing row: {e}")
                 malformed_entity_count += 1
                 continue

        if malformed_entity_count > 0:
            print(f"Warning: Skipped {malformed_entity_count} out of {processed_rows} rows due to malformed 'entities' data or processing errors.")
    else:
        print("Warning: 'entities' or 'text' column not found. Cannot extract initial names.")

    # Final check for empty names before returning
    initial_malware_names = {name for name in initial_malware_names if name}
    print(f"Found {len(initial_malware_names)} unique non-empty potential malware names in 'entities'.")
    return initial_malware_names


def _query_deepseek_for_aliases(malware_name: str) -> list | None:
    # ... (Keep this internal function exactly as before) ...
    if not client: return [] # Return empty list if client not initialized
    if not isinstance(malware_name, str) or not malware_name.strip():
        return [] # Skip empty or invalid names

    prompt = (
        f"List only known aliases or alternate names for the malware \"{malware_name}\". "
        "Return only a comma-separated list (e.g., alias1,alias2,alias three). "
        "If you don't know any aliases or the input is not a malware name, return an empty string. "
        "Do not explain anything or add introductory text. Do not use markdown."
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # Ensure this model name is correct
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=100,
            stream=False,
            timeout=20 # Add a timeout
        )
        # Defensive check for response structure
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
             print(f"  -> DeepSeek returned empty or unexpected response for '{malware_name}'. Skipping.")
             return []

        content = response.choices[0].message.content.strip()

        # More robust check for non-list responses
        if not content or any(p in content.lower() for p in ["i couldn't find", "i do not have information", "no known aliases", "not a malware", "here are some aliases:"]) and ',' not in content:
             # print(f"  -> DeepSeek indicated no aliases for '{malware_name}'.") # Optional debug
             return []

        # Process the comma-separated list
        aliases = [cfg.normalize_name(a) for a in content.split(',') if a.strip()]
        # Filter based on length and block common refusal phrases
        cleaned = [a for a in aliases if len(a) >= cfg.MIN_TERM_LENGTH and not a.startswith("i don't")]

        # if cleaned: print(f"  -> Found aliases for '{malware_name}': {cleaned}") # Optional debug
        return cleaned

    except RateLimitError:
        print(f"  -> Rate limit hit for '{malware_name}'. Waiting...")
        time.sleep(cfg.API_CALL_DELAY_SECONDS * 5) # Exponential backoff might be better
        return None # Indicate failure due to rate limit for potential retry logic
    except APITimeoutError:
         print(f"  -> DeepSeek request timed out for '{malware_name}'. Skipping.")
         return [] # Treat timeout as no aliases found for this attempt
    except APIError as e:
         print(f"  -> DeepSeek API Error for '{malware_name}': {e}. Skipping.")
         return []
    except Exception as e:
        # Catch any other unexpected errors during the API call or processing
        print(f"  -> Unexpected Error during DeepSeek query for '{malware_name}': {type(e).__name__} - {e}. Skipping.")
        return []


def build_malware_dictionary(df: pd.DataFrame) -> tuple[set, bool]:
    """
    Builds the final malware dictionary. Loads from cache if available,
    otherwise extracts initial names, queries DeepSeek, and saves to cache.

    Returns:
        tuple[set, bool]: A tuple containing:
            - The set of malware terms (either loaded or newly built).
            - A boolean indicating if the dictionary was loaded from cache (True) or built fresh (False).
    """
    print("\n--- Building Malware Dictionary ---")
    cache_loaded = False # Flag to indicate if cache was used
    final_dictionary = set()

    # --- Caching Logic ---
    if os.path.exists(cfg.DICTIONARY_CACHE_FILE):
        print(f"Loading malware dictionary from cache file: {cfg.DICTIONARY_CACHE_FILE}")
        try:
            with open(cfg.DICTIONARY_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached_dict = {line.strip() for line in f if line.strip()} # Read, strip, filter empty

            print(f"Successfully loaded {len(cached_dict)} terms from cache.")
            if not cached_dict:
                 print("Warning: Cache file was empty. Proceeding to rebuild dictionary.")
                 # Keep cache_loaded as False
            else:
                 final_dictionary = cached_dict
                 cache_loaded = True # Set flag to True as we are using the cache
                 return final_dictionary, cache_loaded # Return early
        except Exception as e:
            print(f"Error loading dictionary from cache: {e}. Proceeding to rebuild.")
            # Keep cache_loaded as False
    else:
        print("Dictionary cache file not found. Building dictionary from scratch...")
        # Keep cache_loaded as False
    # --- End Caching Logic ---


    # --- Build from scratch if cache wasn't loaded ---
    print("Extracting initial malware names from dataset...")
    initial_names = _extract_initial_malware_names(df)
    if not initial_names:
        print("Warning: No initial malware names extracted. Dictionary will be empty.")
        # Save empty cache file if it doesn't exist to prevent rebuild next time? Optional.
        # Or just return empty set.
        return set(), cache_loaded # Return empty set, cache_loaded is False

    # Add normalized initial names first
    expanded_dict = {cfg.normalize_name(name) for name in initial_names}
    # Filter by minimum length AFTER normalization
    expanded_dict = {name for name in expanded_dict if len(name) >= cfg.MIN_TERM_LENGTH}
    print(f"Added {len(expanded_dict)} initial normalized names (>= {cfg.MIN_TERM_LENGTH} chars).")

    # Query API only if client is available and initialized
    if client:
        print(f"Querying DeepSeek for aliases for {len(initial_names)} unique initial names...")
        api_failures = 0
        successful_queries = 0
        aliases_found_count = 0
        # Use list to ensure order for tqdm, filter names again just in case
        names_to_query = sorted([name for name in initial_names if name and len(name) >= cfg.MIN_TERM_LENGTH])

        for name in tqdm(names_to_query, desc="Querying DeepSeek"):
            aliases = _query_deepseek_for_aliases(name)

            if aliases is None: # Indicates a retryable error like rate limiting
                api_failures += 1
                # Consider adding a retry mechanism here if needed
                continue # Skip to next name for now

            successful_queries += 1
            if aliases: # aliases is an empty list [] if no error but no aliases found
                original_size = len(expanded_dict)
                expanded_dict.update(aliases)
                aliases_found_count += (len(expanded_dict) - original_size)

            # Apply delay even if no aliases found to respect API limits
            time.sleep(cfg.API_CALL_DELAY_SECONDS)

        print(f"DeepSeek querying complete.")
        print(f"  Successful queries: {successful_queries}/{len(names_to_query)}")
        print(f"  API failures (rate limits, timeouts, errors): {api_failures}")
        print(f"  New unique aliases added: {aliases_found_count}")
    else:
        print("DeepSeek client not available or not initialized. Skipping alias expansion.")

    final_dictionary = expanded_dict # Assign the newly built dict
    final_size = len(final_dictionary)
    print(f"🧠 Final dictionary size: {final_size}")

    # --- Save to Cache ---
    # Only save if the dictionary is not empty and it was built fresh (cache_loaded is False)
    if final_dictionary and not cache_loaded:
        print(f"Saving dictionary ({final_size} terms) to cache file: {cfg.DICTIONARY_CACHE_FILE}")
        try:
            # Sort terms for consistent cache file content (easier diffing)
            sorted_terms = sorted(list(final_dictionary))
            with open(cfg.DICTIONARY_CACHE_FILE, 'w', encoding='utf-8') as f:
                for term in sorted_terms:
                    f.write(term + '\n')
            print("Dictionary saved successfully.")
        except Exception as e:
            print(f"Error saving dictionary to cache: {e}")
    elif not final_dictionary:
        print("Warning: Final dictionary is empty. Not saving cache file.")
    # --- End Save to Cache ---

    return final_dictionary, cache_loaded # Return the dict and cache status