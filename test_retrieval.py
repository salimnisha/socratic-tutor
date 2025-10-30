# test_retrieval.py

"""
Test semantic search retrieval
"""

import time
from statistics import mean
from logger_utils import log_data, get_existing_run_id, RETRIEVAL_COLUMNS
from src.retriever import retrieve_relevant_chunks


def test_queries():
    # For logging
    start_time = time.time()
    run_id = get_existing_run_id()

    print("=" * 60)
    print("TESTING SEMANTIC SEARCH")
    print("=" * 60)

    # Test queries
    queries = [
        "Explain the concept of self-supervised learning in machine learning.",
        "What is a token?",
        "Why are foundation models important?",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("+" * 60)

        top_k = 3
        results, metadata_retrieval = retrieve_relevant_chunks(
            query=query, pdf_name="chip_huyen_ch_1", top_k=top_k, return_metadata=True
        )

        similarity_scores = []
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\nResult {i}  (Similarity: {score:.3f})")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
            similarity_scores.append(round(score, 3))

        # logging data
        duration = round((time.time() - start_time), 3)
        avg_similarity = round(mean(similarity_scores), 4)

        log_entry = {
            "run_id": run_id,
            "stage": "test_retrieval",
            "query": query,
            "top_k": top_k,
            "similarity_scores": similarity_scores,
            "avg_similarity": avg_similarity,
            "max_similarity": max(similarity_scores),
            "time_taken_sec": duration,
            "notes": f"Avg score: {avg_similarity}, Time to process: {duration}",
        }

        log_entry.update(metadata_retrieval)
        log_data(log_entry, RETRIEVAL_COLUMNS)


if __name__ == "__main__":
    test_queries()
