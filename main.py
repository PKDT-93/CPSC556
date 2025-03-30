# main.py
"""
Main script to orchestrate the malware detection pipeline:
1. Loads data.
2. Builds/loads malware dictionary & performs dictionary matching (Baseline).
3. **Generates NLP context features (keywords, NER) based on text.**
4. Trains a simple multi-modal classifier (Logistic Regression) using the
   dictionary flag combined with the **NLP context features**.
5. Evaluates both the dictionary-only baseline and the NLP-enhanced classifier.
6. Saves final evaluation results for both approaches to a JSON file.
"""
import sys
import pandas as pd
import ahocorasick
import json
import numpy as np
import datetime
import spacy # Import spaCy
from tqdm.auto import tqdm # For progress bars

# Scikit-learn imports for preprocessing and modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler # Still needed for the combined features
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # Example classifier

# Import from our other modules
import config_utils as cfg
import data_dictionary as dd

# --- spaCy Model Loading ---
# Load the spaCy model once at the beginning for efficiency
NLP_MODEL = None
try:
    # Using the small model for speed, large ('en_core_web_lg') might be more accurate
    NLP_MODEL = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer']) # Disable unused components
    print("spaCy NLP model ('en_core_web_sm') loaded successfully.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    NLP_MODEL = None # Ensure it's None if loading fails
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    NLP_MODEL = None

# --- Matcher Functions (Keep build_automaton and fast_detect as before) ---
def build_automaton(dictionary_terms: set) -> ahocorasick.Automaton | None:
    # ... (keep as before) ...
    print("\n--- Setting up Aho-Corasick Automaton ---")
    if not dictionary_terms:
        print("Warning: Dictionary is empty. Cannot build automaton.")
        return None
    A = ahocorasick.Automaton(ahocorasick.STORE_INTS)
    skipped_terms = 0
    invalid_term_examples = []
    for term in dictionary_terms:
        term_str = str(term).strip()
        if term_str:
            try: A.add_word(term_str, 1)
            except Exception as e:
                 print(f"Error adding term '{term_str}': {e}")
                 skipped_terms += 1
                 if len(invalid_term_examples) < 5: invalid_term_examples.append(repr(term))
        else:
            skipped_terms += 1
            if len(invalid_term_examples) < 5: invalid_term_examples.append(repr(term))
    if skipped_terms > 0:
         print(f"Warning: Skipped {skipped_terms} invalid/empty terms.")
         if invalid_term_examples: print(f"  Examples: {', '.join(invalid_term_examples)}")
    if len(A) == 0:
         print("ERROR: No valid terms added to automaton.")
         return None
    try:
        A.make_automaton()
        print(f"Automaton built successfully with {len(A)} terms.")
        return A
    except Exception as e:
        print(f"ERROR during make_automaton: {e}")
        return None

def fast_detect(text: str, automaton: ahocorasick.Automaton) -> int:
    # ... (keep as before) ...
    if not isinstance(text, str) or not text or automaton is None: return 0
    try:
        first_match = next(automaton.iter(text.lower()), None)
        return 1 if first_match is not None else 0
    except Exception as e:
        print(f"Error during Aho-Corasick search on '{text[:50]}...': {e}")
        return 0

# --- NLP Context Feature Generation Functions ---

def check_info_keywords(text: str) -> int:
    """Checks if any informational keywords are present in the text."""
    if not isinstance(text, str) or not text: return 0
    # Check against lowercased text for case-insensitivity
    text_lower = text.lower()
    for keyword in cfg.INFO_KEYWORDS:
        if keyword in text_lower:
            return 1
    return 0

def check_info_entities(text: str) -> int:
    """Checks if any specified NER entities are present using spaCy."""
    # Return 0 if NLP model isn't loaded or text is invalid
    if NLP_MODEL is None or not isinstance(text, str) or not text:
        return 0
    try:
        doc = NLP_MODEL(text)
        for ent in doc.ents:
            if ent.label_ in cfg.INFO_NER_LABELS:
                return 1 # Found a relevant entity
    except Exception as e:
        print(f"Error during spaCy NER processing on text snippet '{text[:50]}...': {e}")
        return 0 # Return 0 on error
    return 0 # No relevant entity found

# --- Main Pipeline ---
def run_pipeline():
    """Executes the complete malware detection pipeline."""
    start_time = datetime.datetime.now()
    print(f"--- Starting Malware Detection Pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 1. Ensure data is available & Load
    if not dd.download_kaggle_dataset(): sys.exit("Exiting: Dataset download failed.")
    df = dd.load_main_dataframe()
    if df is None or df.empty: sys.exit("Exiting: Dataset loading failed or empty.")
    initial_rows = len(df)
    print(f"Initial dataset rows: {initial_rows}")

    # --- Verify necessary base columns exist ---
    print("Verifying base columns (text, label, entities)...")
    required_base_cols = ['text', 'label', 'entities']
    actual_cols = df.columns.tolist()
    missing_cols = [col for col in required_base_cols if col not in actual_cols]
    if missing_cols:
        print(f"ERROR: Base columns missing: {missing_cols}")
        sys.exit(1)
    print("Base columns found.")

    # 2. Build Dictionary & Perform Dictionary Matching (Baseline Feature)
    malware_dictionary, cache_was_loaded = dd.build_malware_dictionary(df)
    dict_size = len(malware_dictionary)
    automaton = build_automaton(malware_dictionary)
    if automaton is None: sys.exit("Exiting: Automaton build failed.")

    print("\n--- Applying Text Preprocessing and Dictionary Matching (Baseline Feature) ---")
    df['clean_text'] = df['text'].fillna('').astype(str).apply(cfg.preprocess_text)
    print("Applying Aho-Corasick matching to 'clean_text'...")
    tqdm.pandas(desc="Aho-Corasick Matching")
    df[cfg.TEXT_MALWARE_FLAG_FEATURE] = df['clean_text'].progress_apply(lambda txt: fast_detect(txt, automaton))
    print("Dictionary matching complete.")
    print(f"Dictionary flag ('{cfg.TEXT_MALWARE_FLAG_FEATURE}') distribution:\n{df[cfg.TEXT_MALWARE_FLAG_FEATURE].value_counts(normalize=True)}")

    # 3. Generate NLP Context Features
    print("\n--- Generating NLP Context Features ---")
    # Apply keyword check (uses 'clean_text' which is already lowercased/cleaned)
    print("Applying keyword context check...")
    tqdm.pandas(desc="Keyword Check")
    df[cfg.INFO_KEYWORD_FEATURE] = df['clean_text'].progress_apply(check_info_keywords)
    print(f"Info Keyword flag ('{cfg.INFO_KEYWORD_FEATURE}') distribution:\n{df[cfg.INFO_KEYWORD_FEATURE].value_counts(normalize=True)}")

    # Apply NER check (uses original 'text' for better entity recognition)
    # Only run if spaCy model loaded successfully
    if NLP_MODEL:
        print("Applying NER context check (using spaCy)...")
        tqdm.pandas(desc="NER Check")
        # Ensure 'text' is string and fill NaNs before applying
        df[cfg.INFO_ENTITY_FEATURE] = df['text'].fillna('').astype(str).progress_apply(check_info_entities)
        print(f"Info Entity flag ('{cfg.INFO_ENTITY_FEATURE}') distribution:\n{df[cfg.INFO_ENTITY_FEATURE].value_counts(normalize=True)}")
    else:
        print("Skipping NER context check as spaCy model is not loaded.")
        # Create the column with all zeros if NER skipped
        df[cfg.INFO_ENTITY_FEATURE] = 0

    # 4. Prepare Labels
    print("\n--- Preparing True Labels ---")
    df[cfg.TARGET_FEATURE] = df['label'].fillna('benign').astype(str).apply(cfg.map_binary_label)
    print(f"True label ('{cfg.TARGET_FEATURE}') distribution:\n{df[cfg.TARGET_FEATURE].value_counts(normalize=True)}")

    # 5. Split Data (Train/Test)
    print("\n--- Splitting Data into Training and Test Sets ---")
    # ... (keep data splitting logic as before, uses TARGET_FEATURE for stratify) ...
    if len(df) < 2: sys.exit("ERROR: Not enough data for split.")
    if df[cfg.TARGET_FEATURE].nunique() < 2:
        print("Warning: Only one class present. Cannot stratify.")
        train_df, test_df = train_test_split(df, test_size=cfg.TEST_SET_SIZE, random_state=cfg.RANDOM_STATE)
    else:
        try:
            train_df, test_df = train_test_split(
                df, test_size=cfg.TEST_SET_SIZE, random_state=cfg.RANDOM_STATE, stratify=df[cfg.TARGET_FEATURE]
            )
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}). Using non-stratified split.")
            train_df, test_df = train_test_split(df, test_size=cfg.TEST_SET_SIZE, random_state=cfg.RANDOM_STATE)
    if test_df.empty: sys.exit("ERROR: Test set is empty after splitting.")
    print(f"Dataset Split: {len(train_df)} training rows / {len(test_df)} test rows")
    print(f"Test set label distribution:\n{test_df[cfg.TARGET_FEATURE].value_counts(normalize=True)}")

    # Define target variables
    y_train = train_df[cfg.TARGET_FEATURE]
    y_test = test_df[cfg.TARGET_FEATURE]

    # 6. Evaluate Dictionary-Only Baseline
    print("\n--- Evaluating Dictionary-Only Baseline on Test Set ---")
    y_pred_baseline = test_df[cfg.TEXT_MALWARE_FLAG_FEATURE]
    # ... (keep baseline evaluation logic as before) ...
    conf_matrix_baseline = None; report_dict_baseline = {}; report_str_baseline = "N/A"
    try:
        conf_matrix_baseline = confusion_matrix(y_test, y_pred_baseline)
        target_names = ['Not Malware (0)', 'Malware (1)']
        report_dict_baseline = classification_report(y_test, y_pred_baseline, target_names=target_names, zero_division=0, output_dict=True)
        report_str_baseline = classification_report(y_test, y_pred_baseline, target_names=target_names, zero_division=0)
        print("\nBaseline Confusion Matrix:\n", conf_matrix_baseline)
        print("\nBaseline Classification Report:\n", report_str_baseline)
    except Exception as e:
        print(f"Error during baseline evaluation: {e}")
        report_dict_baseline = {"error": str(e)}

    # 7. Prepare Features and Train NLP-Enhanced Classifier
    print("\n--- Preparing Features for NLP-Enhanced Classifier ---")
    # Define the features to use: dictionary flag + new context features
    context_features_used = [cfg.TEXT_MALWARE_FLAG_FEATURE, cfg.INFO_KEYWORD_FEATURE]
    if NLP_MODEL: # Only include NER feature if spaCy ran
        context_features_used.append(cfg.INFO_ENTITY_FEATURE)
    print(f"Features used for classifier: {context_features_used}")

    # Select feature columns from train/test sets
    X_train_context = train_df[context_features_used].values
    X_test_context = test_df[context_features_used].values
    print(f"Context feature shapes: Train={X_train_context.shape}, Test={X_test_context.shape}")

    # --- Optional Scaling ---
    # Scaling might be beneficial even for binary features if using regularization
    print("Scaling context features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_context)
    X_test_scaled = scaler.transform(X_test_context)
    # --- End Optional Scaling ---

    # Train and Evaluate NLP-Enhanced Classifier
    model = None; conf_matrix_model = None; report_dict_model = {}; report_str_model = "N/A"
    if X_train_scaled is not None and X_test_scaled is not None:
        print("\n--- Training NLP-Enhanced Classifier (Logistic Regression) ---")
        model = LogisticRegression(
            random_state=cfg.RANDOM_STATE,
            max_iter=1000,
            class_weight='balanced', # Important for imbalance
            solver='liblinear'
        )
        try:
            model.fit(X_train_scaled, y_train)
            print("Model training complete.")

            print("\n--- Evaluating NLP-Enhanced Classifier on Test Set ---")
            y_pred_model = model.predict(X_test_scaled)
            conf_matrix_model = confusion_matrix(y_test, y_pred_model)
            report_dict_model = classification_report(y_test, y_pred_model, target_names=target_names, zero_division=0, output_dict=True)
            report_str_model = classification_report(y_test, y_pred_model, target_names=target_names, zero_division=0)
            print("\nNLP-Enhanced Model Confusion Matrix:\n", conf_matrix_model)
            print("\nNLP-Enhanced Model Classification Report:\n", report_str_model)
        except Exception as e:
            print(f"ERROR during NLP-Enhanced model training or evaluation: {e}")
            report_dict_model = {"error": f"Model training/prediction failed: {e}"}
            report_str_model = f"N/A (Error: {e})"
    else:
        print("Skipping NLP-Enhanced model training due to feature preparation issues.")
        report_dict_model = {"error": "Skipped due to feature prep issues"}

    # 8. Save Combined Results to JSON File
    print("\n--- Saving Combined Results ---")
    end_time = datetime.datetime.now()
    results_filename = "pipeline_results_nlp_context.json" # New filename

    results_data = {
        "run_timestamp_start": start_time.isoformat(),
        "run_timestamp_end": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        "config": {
            "test_set_size": cfg.TEST_SET_SIZE,
            "random_state": cfg.RANDOM_STATE,
            "min_term_length": cfg.MIN_TERM_LENGTH,
            "dictionary_cache_file": cfg.DICTIONARY_CACHE_FILE,
            "deepseek_enabled": dd.is_deepseek_client_active(),
            "nlp_context_features_config": { # Log NLP config
                "info_keywords": cfg.INFO_KEYWORDS,
                "info_ner_labels": cfg.INFO_NER_LABELS,
                "spacy_model_used": "en_core_web_sm" if NLP_MODEL else "None",
            },
            "classifier_used": model.__class__.__name__ if model is not None else "N/A",
        },
        "dataset_info": {
            "csv_path": cfg.LOCAL_CSV_PATH,
            "initial_rows": initial_rows,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "full_set_label_dist": df[cfg.TARGET_FEATURE].value_counts(normalize=True).to_dict(),
        },
        "dictionary_info": {
             "size": dict_size,
             "loaded_from_cache": cache_was_loaded,
        },
        "dictionary_only_baseline_evaluation": {
            "confusion_matrix": conf_matrix_baseline.tolist() if isinstance(conf_matrix_baseline, np.ndarray) else conf_matrix_baseline,
            "classification_report_dict": report_dict_baseline,
            "classification_report_string": report_str_baseline
        },
        "nlp_enhanced_classifier_evaluation": { # Renamed section
            "features_used": context_features_used,
            "confusion_matrix": conf_matrix_model.tolist() if isinstance(conf_matrix_model, np.ndarray) else conf_matrix_model,
            "classification_report_dict": report_dict_model,
            "classification_report_string": report_str_model
        }
    }

    try:
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, default=str)
        print(f"NLP context comparison results saved successfully to: {results_filename}")
    except Exception as e:
        print(f"ERROR: Failed to save NLP context results to {results_filename}: {e}")

    print("\n--- Pipeline Finished ---")

# --- Execution Guard ---
if __name__ == "__main__":
    run_pipeline()