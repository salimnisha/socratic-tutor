# logger_utils.py

"""
Log into csv and json files
"""

import csv
import json
import inspect
import uuid
from datetime import datetime
from pathlib import Path

PROCESS_PDF_COLUMNS = [
    "run_id",
    "timestamp",
    "stage",
    # Inputs
    "pdf_name",
    # Processing
    "chunking_method",
    "chunk_size",
    "overlap",
    "num_chunks",
    "avg_chunk_size_created",
    # Tokens and cost
    "num_embeddings",
    "embedding_tokens",
    "avg_tokens_per_embedding",
    "embedding_cost_usd",
    # Model
    "embedding_model",
    # Time
    "time_taken_sec",
    # Notes
    "notes",
]

LOG_COLUMN_ORDER = [
    # Identifiers
    "run_id",
    "timestamp",
    "stage",
    # Inputs
    "pdf_name",
    "query",
    # Processing
    "chunking_method",
    "num_chunks",
    "chunk_size",
    "overlap",
    "avg_chunk_size_created",
    "top_k",
    # Similarity
    "max_similarity",
    "avg_similarity",
    "all_top3_scores",
    "avg_top3_score",
    # Tokens and cost
    "num_embeddings",
    "embedding_tokens",
    "avg_tokens_per_embedding",
    "gpt_input_tokens",
    "gpt_output_tokens",
    "total_tokens",
    "embedding_cost_usd",
    "gpt_cost_usd",
    "total_cost_usd",
    # Timing
    "time_taken_sec",
    # Model
    "embedding_model",
    "temperature",
    # Notes
    "notes",
]


# ------------------------------------------------
# Run ID Management
# ------------------------------------------------
def create_new_run_id():
    """Create a new run_id and save to file"""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    Path("logs").mkdir(exist_ok=True)
    (Path("logs") / "current_run_id.txt").write_text(run_id)
    return run_id


def get_existing_run_id():
    """Retrieve the current run id (if it exists)"""
    run_file = Path("logs") / "current_run_id.txt"
    return run_file.read_text().strip() if run_file.exists else None


# ------------------------------------------------
# Logging
# ------------------------------------------------
def log_data(log_entry, column_order):
    """
    Simple log file that appends to csv and jsonl
    Autofills timestamp and a short summary note
    Autodetects the caller script to choose the logfile (e.g., if called from process_pdf, log file will be process_pdf.csv)

    Args:
        log_entry::dict
            A dictionary that is populated with values to log
        column_order::list
            The order of the headers in csv file
    """

    # Detect the caller script
    caller = inspect.stack()[1].filename.lower()
    if "process_pdf" in caller:
        logfile_name = "processing"
    elif "test_retrieval" in caller:
        logfile_name = "retrieval"
    else:
        logfile_name = "other"

    # Set up paths
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    csv_path = logs_dir / f"{logfile_name}.csv"
    jsonl_path = csv_path.with_suffix(".jsonl")

    # Add timestamp
    log_entry.setdefault("timestamp", datetime.now().isoformat())

    # CSV append
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=column_order, extrasaction="ignore"
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    # JSONL append
    with open(jsonl_path, "a") as json_file:
        json_file.write(json.dumps(log_entry) + "\n")

    print(f"✅ Logged -> {csv_path}")
