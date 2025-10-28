# test_qa.py

"""
Test the complete Q&A system
"""

from src.qa_system import answer_query

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

for question in questions:
    answer = answer_query(
        question, pdf_name="chip_huyen_ch_1", top_k=3, show_context=True
    )
    print("\n" + "=" * 60 + "\n")
