# test_qa.py

"""
Test the complete Q&A system
"""

from src.qa_system import answer_query
from logger_utils import QA_COLUMNS, get_existing_run_id, log_data

# Test questions
questions = [
    "What is self-supervision in machine learning?",
    "What is a token in language models?",
    "How do foundation models differ from traditional ML models?",
    "What is the meaning of life?",  # Should say "not in material"
]

print("=" * 60)
print("TESTING Q&A system")
print("=" * 60)

run_id = get_existing_run_id()

for question in questions:
    answer, metadata_qa = answer_query(
        question,
        pdf_name="chip_huyen_ch_1",
        top_k=3,
        show_context=True,
        return_metadata=True,
    )

    metadata = {
        "run_id": run_id,
        "stage": "test_qa",
    }
    metadata.update(metadata_qa)

    log_data(metadata, QA_COLUMNS)

    print("\n" + "=" * 60 + "\n")
