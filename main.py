# main.py
"""
Main script to orchestrate the malware detection pipeline:
1. Loads data (processed format with relations).
2. Builds/loads malware dictionary & performs dictionary matching (Baseline).
3. Generates NLP context features (keywords, NER).
4. **Parses 'relations' and generates Graph-based features.**
5. Derives the true binary label from flattened entity labels.
6. Trains and evaluates three approaches:
    a) Dictionary-Only Baseline
    b) NLP-Context-Enhanced Classifier
    c) Graph-Enhanced Classifier (using dict flag + NLP context + graph features)
7. Saves final evaluation results for all approaches to a JSON file.
**Updated for Graph Feature Integration**
"""
import sys
import pandas as pd
import ahocorasick
import json
import numpy as np
import datetime
import spacy
import re
import ast # For safely parsing 'relations' string
import networkx as nx # For graph operations
from tqdm.auto import tqdm

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Import from our other modules
import config_utils as cfg
import data_dictionary as dd

# --- spaCy Model Loading (Remains the same) ---
NLP_MODEL = None
try:
    NLP_MODEL = spacy.load("en_core_web_sm", disable=['parser', 'lemmatizer'])
    print("spaCy NLP model ('en_core_web_sm') loaded successfully.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    NLP_MODEL = None
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    NLP_MODEL = None

# --- Matcher Functions (Remain the same) ---
def build_automaton(dictionary_terms: set) -> ahocorasick.Automaton | None:
    # Function remains the same as previous version
    print("\n--- Setting up Aho-Corasick Automaton ---")
    if not dictionary_terms: print("Warning: Dictionary is empty. Cannot build automaton."); return None
    A = ahocorasick.Automaton(ahocorasick.STORE_INTS); skipped_terms = 0; invalid_term_examples = []
    for term in dictionary_terms:
        term_str = str(term).strip()
        if term_str:
            try: A.add_word(term_str, 1)
            except Exception as e: skipped_terms += 1; # if len(invalid_term_examples) < 5: invalid_term_examples.append(repr(term))
        else: skipped_terms += 1; # if len(invalid_term_examples) < 5: invalid_term_examples.append(repr(term))
    if skipped_terms > 0: print(f"Warning: Skipped {skipped_terms} invalid/empty terms during automaton build.")
    if len(A) == 0: print("ERROR: No valid terms added to automaton."); return None
    try: A.make_automaton(); print(f"Automaton built successfully with {len(A)} terms."); return A
    except Exception as e: print(f"ERROR during make_automaton: {e}"); return None

def fast_detect(text: str, automaton: ahocorasick.Automaton) -> int:
    # Function remains the same as previous version
    if not isinstance(text, str) or not text or automaton is None: return 0
    try: first_match = next(automaton.iter(text.lower()), None); return 1 if first_match is not None else 0
    except Exception as e: return 0

# --- NLP Context Feature Generation Functions (Remain the same) ---
def check_info_keywords(text: str) -> int:
    # Function remains the same as previous version
    if not isinstance(text, str) or not text: return 0
    for keyword in cfg.INFO_KEYWORDS:
        if keyword in text: return 1
    return 0

def check_info_entities(text: str) -> int:
    # Function remains the same as previous version
    if NLP_MODEL is None or not isinstance(text, str) or not text: return 0
    try:
        doc = NLP_MODEL(text)
        for ent in doc.ents:
            if ent.label_ in cfg.INFO_NER_LABELS: return 1
    except Exception as e: return 0
    return 0

# --- Graph Feature Extraction Functions
def parse_relations(relations_str: str) -> list[tuple[int, int]]:
    """
    Safely parses the string representation of relations list,
    handling a list of dictionaries with 'from_id' and 'to_id' keys.
    Returns a list of (source_id, target_id) integer tuples.
    """
    if not isinstance(relations_str, str) or not relations_str.strip():
        return []
    try:
        # Use literal_eval for safe evaluation of Python literals
        relations_list = ast.literal_eval(relations_str)

        # Validate if it's a list
        if not isinstance(relations_list, list):
            # print(f"Warning: Parsed relations is not a list: {type(relations_list)}") # Debug
            return []

        valid_relations = []
        for item in relations_list:
            # Check if item is a dictionary and has the required keys
            if isinstance(item, dict) and 'from_id' in item and 'to_id' in item:
                try:
                    # Extract IDs using dictionary keys, convert to int
                    id1 = int(item['from_id'])
                    id2 = int(item['to_id'])
                    valid_relations.append((id1, id2))
                except (ValueError, TypeError, KeyError):
                    # print(f"Warning: Could not extract valid integer IDs from relation dict: {item}") # Debug
                    continue # Skip item if IDs are missing, not convertible, or keys wrong
            # else: # Debug
                # print(f"Warning: Skipping relation item that is not a dict or lacks keys: {item}")

        return valid_relations
    except (ValueError, SyntaxError, TypeError):
        # print(f"Warning: Failed to parse relations string: {relations_str[:100]}...") # Debug
        return [] # Return empty list on parsing error
    except Exception as e: # Catch any other unexpected errors
        # print(f"Warning: Unexpected error parsing relations: {e}") # Debug
        return []

def extract_graph_features(row: pd.Series, max_entity_index: int) -> pd.Series:
    """
    Extracts graph-based features for a given row.
    Features: max degree of malware nodes, total edges.
    """
    # Default feature values
    max_malware_degree = 0
    total_edges = 0

    # 1. Parse relations
    relations = parse_relations(row.get('relations', '')) # Use .get for safety
    total_edges = len(relations)

    if not relations:
        # Return default values if no relations or parsing failed
        return pd.Series([max_malware_degree, total_edges], index=[cfg.MAX_MALWARE_DEGREE_FEATURE, cfg.TOTAL_EDGES_FEATURE])

    # 2. Build graph
    try:
        G = nx.Graph() # Undirected graph
        G.add_edges_from(relations)
    except Exception as e:
        # print(f"Warning: Error building graph for row {row.name}: {e}") # Debug
        # Return default values if graph building fails
        return pd.Series([max_malware_degree, total_edges], index=[cfg.MAX_MALWARE_DEGREE_FEATURE, cfg.TOTAL_EDGES_FEATURE])
    print(f"Row {row.name}: Graph Edges: {list(G.edges())}")
    print(f"Row {row.name}: Graph Nodes: {list(G.nodes())}")
    # 3. Find malware nodes and calculate max degree
    malware_nodes_in_graph = set()
    for i in range(1, max_entity_index + 1):
        label_col = f'label_{i}'
        id_col = f'id_{i}'
        # Check if columns exist and label is malware
        if label_col in row and id_col in row and \
           isinstance(row[label_col], str) and row[label_col].lower() == 'malware':
            try:
                entity_id = int(row[id_col]) # Ensure ID is integer
                print(f"Entity ID {entity_id}")
                # Check if this entity ID is actually a node in the graph derived from relations
                print(f"G has_node entity ID {G.has_node(entity_id)}")
                if G.has_node(entity_id):
                    malware_nodes_in_graph.add(entity_id)
            except (ValueError, TypeError):
                continue # Skip if id is not convertible to int

    # Calculate max degree among identified malware nodes
    if malware_nodes_in_graph:
        max_malware_degree = 0
        for node_id in malware_nodes_in_graph:
            try:
                degree = G.degree(node_id)
                max_malware_degree = max(max_malware_degree, degree)
            except Exception as e:
                # print(f"Warning: Error getting degree for node {node_id} in row {row.name}: {e}") # Debug
                continue # Skip node if degree calculation fails

    return pd.Series([max_malware_degree, total_edges], index=[cfg.MAX_MALWARE_DEGREE_FEATURE, cfg.TOTAL_EDGES_FEATURE])


# --- True Label Derivation Function (Remains the same) ---
def derive_true_label(row: pd.Series, max_entity_index: int) -> int:
    for i in range(1, max_entity_index + 1):
        label_col = f'label_{i}'
        if label_col in row:
            label = row[label_col]
            if isinstance(label, str) and label.lower() == 'malware':
                return 1
    return 0

# --- Main Pipeline ---
def run_pipeline():
    start_time = datetime.datetime.now()
    print(f"--- Starting Malware Detection Pipeline (Graph Enhanced) at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 1. Load Data (checks for text, relations, id_1, label_1, etc.)
    if not dd.download_kaggle_dataset():
         print("Warning: Kaggle download step indicated failure or was skipped.")
         if not os.path.exists(cfg.LOCAL_CSV_PATH): sys.exit(f"Exiting: Dataset file not found at {cfg.LOCAL_CSV_PATH}.")
    df = dd.load_main_dataframe()
    if df is None or df.empty: sys.exit("Exiting: Dataset loading failed or empty.")
    initial_rows = len(df)
    print(f"Initial dataset rows: {initial_rows}")

    # --- Find max entity index (needed for label derivation and graph features) ---
    max_entity_index = 0
    for col in df.columns:
        match = re.match(r'label_(\d+)', col)
        if match: max_entity_index = max(max_entity_index, int(match.group(1)))
    if max_entity_index == 0: sys.exit("ERROR: Could not determine maximum entity index from columns.")
    print(f"Determined maximum entity index: {max_entity_index}")

    # 2. Build Dictionary & Perform Dictionary Matching (Baseline Feature)
    malware_dictionary, cache_was_loaded = dd.build_malware_dictionary(df)
    dict_size = len(malware_dictionary)
    automaton = build_automaton(malware_dictionary)
    if automaton is None: sys.exit("Exiting: Automaton build failed.")

    print("\n--- Applying Text Preprocessing and Dictionary Matching ---")
    df['clean_text'] = df['text'].fillna('').astype(str).apply(cfg.preprocess_text)
    tqdm.pandas(desc="Aho-Corasick Matching")
    df[cfg.TEXT_MALWARE_FLAG_FEATURE] = df['clean_text'].progress_apply(lambda txt: fast_detect(txt, automaton))
    print(f"'{cfg.TEXT_MALWARE_FLAG_FEATURE}' distribution:\n{df[cfg.TEXT_MALWARE_FLAG_FEATURE].value_counts(normalize=True)}")

    # 3. Generate NLP Context Features
    print("\n--- Generating NLP Context Features ---")
    tqdm.pandas(desc="Keyword Check")
    df[cfg.INFO_KEYWORD_FEATURE] = df['clean_text'].progress_apply(check_info_keywords)
    print(f"'{cfg.INFO_KEYWORD_FEATURE}' distribution:\n{df[cfg.INFO_KEYWORD_FEATURE].value_counts(normalize=True)}")
    if NLP_MODEL:
        tqdm.pandas(desc="NER Check")
        df[cfg.INFO_ENTITY_FEATURE] = df['text'].fillna('').astype(str).progress_apply(check_info_entities)
        print(f"'{cfg.INFO_ENTITY_FEATURE}' distribution:\n{df[cfg.INFO_ENTITY_FEATURE].value_counts(normalize=True)}")
    else:
        print("Skipping NER context check as spaCy model is not loaded.")
        df[cfg.INFO_ENTITY_FEATURE] = 0

    # 4. *** Generate Graph Features ***
    print("\n--- Generating Graph Features ---")
    tqdm.pandas(desc="Graph Feature Extraction")
    graph_features_df = df.progress_apply(lambda row: extract_graph_features(row, max_entity_index), axis=1)
    df = pd.concat([df, graph_features_df], axis=1)
    print("Graph feature extraction complete.")
    print(f"'{cfg.MAX_MALWARE_DEGREE_FEATURE}' distribution summary:\n{df[cfg.MAX_MALWARE_DEGREE_FEATURE].describe()}")
    print(f"'{cfg.TOTAL_EDGES_FEATURE}' distribution summary:\n{df[cfg.TOTAL_EDGES_FEATURE].describe()}")

    # 5. Prepare True Labels
    print("\n--- Preparing True Labels ---")
    tqdm.pandas(desc="Deriving True Label")
    df[cfg.TARGET_FEATURE] = df.progress_apply(lambda row: derive_true_label(row, max_entity_index), axis=1)
    print(f"Derived '{cfg.TARGET_FEATURE}' distribution:\n{df[cfg.TARGET_FEATURE].value_counts(normalize=True)}")

    # 6. Split Data
    print("\n--- Splitting Data into Training and Test Sets ---")
    # ... (Splitting logic remains the same, uses derived TARGET_FEATURE) ...
    if len(df) < 2: sys.exit("ERROR: Not enough data for split.")
    if df[cfg.TARGET_FEATURE].nunique() < 2:
        print(f"Warning: Only one class ({df[cfg.TARGET_FEATURE].unique()}) present in derived labels. Cannot stratify or train effectively.")
        train_df, test_df = train_test_split(df, test_size=cfg.TEST_SET_SIZE, random_state=cfg.RANDOM_STATE)
    else:
        try: train_df, test_df = train_test_split(df, test_size=cfg.TEST_SET_SIZE, random_state=cfg.RANDOM_STATE, stratify=df[cfg.TARGET_FEATURE])
        except ValueError as e: print(f"Warning: Stratified split failed ({e}). Using non-stratified split."); train_df, test_df = train_test_split(df, test_size=cfg.TEST_SET_SIZE, random_state=cfg.RANDOM_STATE)
    if test_df.empty: sys.exit("ERROR: Test set is empty after splitting.")
    print(f"Dataset Split: {len(train_df)} training rows / {len(test_df)} test rows")
    print(f"Test set derived label distribution:\n{test_df[cfg.TARGET_FEATURE].value_counts(normalize=True)}")
    y_train = train_df[cfg.TARGET_FEATURE]
    y_test = test_df[cfg.TARGET_FEATURE]

    # --- 7. Evaluate Models ---

    # a) Dictionary-Only Baseline
    print("\n--- Evaluating Dictionary-Only Baseline on Test Set ---")
    y_pred_baseline = test_df[cfg.TEXT_MALWARE_FLAG_FEATURE]
    # ... (Baseline evaluation logic remains the same) ...
    conf_matrix_baseline = None; report_dict_baseline = {}; report_str_baseline = "N/A"
    try:
        conf_matrix_baseline = confusion_matrix(y_test, y_pred_baseline)
        target_names = ['Not Malware (0)', 'Malware (1)']
        report_dict_baseline = classification_report(y_test, y_pred_baseline, target_names=target_names, zero_division=0, output_dict=True)
        report_str_baseline = classification_report(y_test, y_pred_baseline, target_names=target_names, zero_division=0)
        print("\nBaseline Confusion Matrix:\n", conf_matrix_baseline)
        print("\nBaseline Classification Report:\n", report_str_baseline)
    except Exception as e: print(f"Error during baseline evaluation: {e}"); report_dict_baseline = {"error": str(e)}


    # b) NLP-Context-Enhanced Classifier
    print("\n--- Evaluating NLP-Context-Enhanced Classifier ---")
    nlp_context_features = [cfg.TEXT_MALWARE_FLAG_FEATURE, cfg.INFO_KEYWORD_FEATURE]
    if NLP_MODEL: nlp_context_features.append(cfg.INFO_ENTITY_FEATURE)
    print(f"Features used: {nlp_context_features}")
    X_train_nlp = train_df[nlp_context_features].values
    X_test_nlp = test_df[nlp_context_features].values
    model_nlp = None; conf_matrix_nlp = None; report_dict_nlp = {}; report_str_nlp = "N/A"
    if X_train_nlp.shape[0] > 0 and y_train.nunique() > 1:
        scaler_nlp = StandardScaler()
        X_train_nlp_scaled = scaler_nlp.fit_transform(X_train_nlp)
        X_test_nlp_scaled = scaler_nlp.transform(X_test_nlp)
        model_nlp = LogisticRegression(random_state=cfg.RANDOM_STATE, max_iter=1000, class_weight='balanced', solver='liblinear')
        try:
            model_nlp.fit(X_train_nlp_scaled, y_train)
            y_pred_nlp = model_nlp.predict(X_test_nlp_scaled)
            conf_matrix_nlp = confusion_matrix(y_test, y_pred_nlp)
            report_dict_nlp = classification_report(y_test, y_pred_nlp, target_names=target_names, zero_division=0, output_dict=True)
            report_str_nlp = classification_report(y_test, y_pred_nlp, target_names=target_names, zero_division=0)
            print("\nNLP-Context-Enhanced Model Confusion Matrix:\n", conf_matrix_nlp)
            print("\nNLP-Context-Enhanced Model Classification Report:\n", report_str_nlp)
        except Exception as e: print(f"ERROR during NLP-Context model training/eval: {e}"); report_dict_nlp = {"error": str(e)}
    else: print("Skipping NLP-Context model training (no data or single class)."); report_dict_nlp = {"error": "Skipped"}


    # c) Graph-Enhanced Classifier (Dict Flag + NLP Context + Graph Features)
    print("\n--- Evaluating Graph-Enhanced Classifier ---")
    graph_enhanced_features = nlp_context_features + [cfg.MAX_MALWARE_DEGREE_FEATURE, cfg.TOTAL_EDGES_FEATURE]
    print(f"Features used: {graph_enhanced_features}")
    # Select features, handling potential NaNs from graph extraction (fillna with 0?)
    X_train_graph = train_df[graph_enhanced_features].fillna(0).values
    X_test_graph = test_df[graph_enhanced_features].fillna(0).values
    model_graph = None; conf_matrix_graph = None; report_dict_graph = {}; report_str_graph = "N/A"
    if X_train_graph.shape[0] > 0 and y_train.nunique() > 1:
        scaler_graph = StandardScaler() # Scale all features together
        X_train_graph_scaled = scaler_graph.fit_transform(X_train_graph)
        X_test_graph_scaled = scaler_graph.transform(X_test_graph)
        model_graph = LogisticRegression(random_state=cfg.RANDOM_STATE, max_iter=1000, class_weight='balanced', solver='liblinear')
        try:
            model_graph.fit(X_train_graph_scaled, y_train)
            y_pred_graph = model_graph.predict(X_test_graph_scaled)
            conf_matrix_graph = confusion_matrix(y_test, y_pred_graph)
            report_dict_graph = classification_report(y_test, y_pred_graph, target_names=target_names, zero_division=0, output_dict=True)
            report_str_graph = classification_report(y_test, y_pred_graph, target_names=target_names, zero_division=0)
            print("\nGraph-Enhanced Model Confusion Matrix:\n", conf_matrix_graph)
            print("\nGraph-Enhanced Model Classification Report:\n", report_str_graph)
        except Exception as e: print(f"ERROR during Graph-Enhanced model training/eval: {e}"); report_dict_graph = {"error": str(e)}
    else: print("Skipping Graph-Enhanced model training (no data or single class)."); report_dict_graph = {"error": "Skipped"}


    # 8. Save Combined Results
    print("\n--- Saving Combined Results ---")
    end_time = datetime.datetime.now()
    results_filename = "pipeline_results_graph_enhanced.json" # New filename

    results_data = {
        "run_timestamp_start": start_time.isoformat(),
        "run_timestamp_end": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        "config": {
            "dataset_csv": cfg.MAIN_CSV_FILE,
            "test_set_size": cfg.TEST_SET_SIZE,
            "random_state": cfg.RANDOM_STATE,
            "min_term_length": cfg.MIN_TERM_LENGTH,
            "dictionary_cache_file": cfg.DICTIONARY_CACHE_FILE,
            "deepseek_enabled": dd.is_deepseek_client_active(),
            "spacy_model_loaded": NLP_MODEL is not None,
            "true_label_derivation": f"Derived: 1 if 'malware' in label_1..{max_entity_index}, else 0",
        },
        "dataset_info": {
            "initial_rows": initial_rows, "train_rows": len(train_df), "test_rows": len(test_df),
            "full_set_derived_label_dist": df[cfg.TARGET_FEATURE].value_counts(normalize=True).to_dict(),
        },
        "dictionary_info": {"size": dict_size, "loaded_from_cache": cache_was_loaded},
        "dictionary_only_baseline_evaluation": {
            "confusion_matrix": conf_matrix_baseline.tolist() if isinstance(conf_matrix_baseline, np.ndarray) else conf_matrix_baseline,
            "classification_report_dict": report_dict_baseline,
            "classification_report_string": report_str_baseline
        },
        "nlp_context_enhanced_evaluation": { # Added this section
            "features_used": nlp_context_features,
            "confusion_matrix": conf_matrix_nlp.tolist() if isinstance(conf_matrix_nlp, np.ndarray) else conf_matrix_nlp,
            "classification_report_dict": report_dict_nlp,
            "classification_report_string": report_str_nlp
        },
         "graph_enhanced_evaluation": { # Added this section
            "features_used": graph_enhanced_features,
            "confusion_matrix": conf_matrix_graph.tolist() if isinstance(conf_matrix_graph, np.ndarray) else conf_matrix_graph,
            "classification_report_dict": report_dict_graph,
            "classification_report_string": report_str_graph
        }
    }

    try:
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, default=str)
        print(f"Results saved successfully to: {results_filename}")
    except Exception as e:
        print(f"ERROR: Failed to save results to {results_filename}: {e}")

    print("\n--- Pipeline Finished ---")

# --- Execution Guard ---
if __name__ == "__main__":
    # Ensure networkx is installed: pip install networkx
    run_pipeline()