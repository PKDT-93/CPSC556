# CPSC 556 Malware Detection Pipeline using Dictionary Matching

This project implements a pipeline to detect potential malware references in text data using a dictionary-based approach with Aho-Corasick matching. It includes steps for data loading (from a Kaggle dataset), dictionary building (optionally enhanced with an LLM like DeepSeek), caching the dictionary, text preprocessing, matching, and evaluation.

## Prerequisites

1.  **Python:** Python 3.8 or higher is recommended.
2.  **Libraries:** Install required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Ensure your `requirements.txt` file includes at least:
    *   `pandas`
    *   `pyahocorasick`
    *   `scikit-learn`
    *   `openai` (if using DeepSeek/LLM expansion)
    *   `python-dotenv` (if using DeepSeek/LLM expansion)
    *   `tqdm` (for progress bars)
    *   `kaggle` (optional, for automatic dataset download)

3.  **Kaggle API (Optional but Recommended):**
    *   If you want the script to automatically download the dataset, you need the Kaggle API tool installed and configured.
    *   Install: `pip install kaggle`
    *   Configure: Download your `kaggle.json` API token from your Kaggle account settings and place it in `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<YourUsername>\.kaggle\kaggle.json` (Windows). Ensure the file has appropriate permissions (read/write for user only).

4.  **DeepSeek API Key (Optional, HIGHLY RECOMMENDED):**
    *   If you want to use the LLM (DeepSeek) to expand the malware dictionary with aliases:
        *   Create a file named `.env` in the same directory as `main.py`.
        *   Add your DeepSeek API key to the `.env` file like this:
            ```dotenv
            DEEPSEEK_API_KEY=your_actual_api_key_here
            ```
    *   If you don't provide a key (or the `.env` file), the script will skip the LLM expansion step and rely only on names extracted from the dataset's `entities` column (or the cache if it exists).

## Folder Structure

Ensure your project directory looks like this:
```
CPSC556/
├── main.py
├── data_dictionary.py
├── config_utils.py
├── requirements.txt
├── .env # Optional: For DeepSeek API Key
└── kaggle_data/ # Created automatically or manually place dataset here
└── cyber-threat-intelligence_all.csv # Dataset file
```

The `kaggle_data` directory will be created by the script if it doesn't exist and the Kaggle download is successful. The `cyber-threat-intelligence_all.csv` file should reside inside it.

## How to Run

1.  Navigate to the project's root directory (`CPSC556/`) in your terminal.
2.  Make sure prerequisites are met (libraries installed, optional `.env` and Kaggle setup done).
3.  Execute the main script:
    ```bash
    python main.py
    ```

## Expected Outputs

When you run the script, you will see output messages in your console indicating the progress of each step (data loading, dictionary building/loading, matching, evaluation).

Two key files will be generated (or updated) in the root directory (`CPSC556/`):

1.  **`malware_dictionary_cache.txt`:**
    *   **First Run:** If this file doesn't exist, the script builds the dictionary (extracting names, optionally querying DeepSeek) and saves the resulting terms (one per line) into this file.
    *   **Subsequent Runs:** If this file *does* exist, the script loads the dictionary terms directly from it, significantly speeding up the startup by skipping the extraction and LLM calls.
    *   **Content:** A list of normalized malware terms used by the Aho-Corasick automaton.

2.  **`pipeline_results.json`:**
    *   **Content:** This file contains a structured summary of the pipeline's execution and the final evaluation results calculated on the test set. It includes:
        *   Run timestamps and duration.
        *   Key configuration parameters used.
        *   Dataset information (rows, splits).
        *   Dictionary information (size, whether it was loaded from cache).
        *   **Evaluation results:** Confusion Matrix and Classification Report (both as a dictionary and a formatted string).
    *   **Purpose:** This is the **primary file to reference for your report's results section**. Its JSON format makes it easy to parse programmatically or feed into an LLM for report generation.

## Interpreting Results

*   Check the **console output** for any warnings or errors during execution.
*   Open `pipeline_results.json`. The key section for performance is `evaluation_on_test_set`. Look at the `confusion_matrix` and `classification_report_dict` fields for detailed metrics (precision, recall, F1-score) for the 'Malware' and 'Not Malware' classes.

## Demonstrating Caching

To see the caching mechanism in action:

1.  **Delete** the `malware_dictionary_cache.txt` file if it exists.
2.  Run `python main.py`. Observe the console output showing "Building dictionary from scratch..." and potentially "Querying DeepSeek...". Note the time taken.
3.  Run `python main.py` **again** *without* deleting the cache file. Observe the console output now showing "Loading malware dictionary from cache file...". The script should start the matching phase much faster.# CPSC556
