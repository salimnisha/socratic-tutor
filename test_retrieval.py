# test_retrieval.py

"""
Test semantic search retrieval
"""

from src.retriever import retrieve_relevant_chunks


def test_queries():
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

        results = retrieve_relevant_chunks(
            query=query, pdf_name="chip_huyen_ch_1", top_k=3
        )

        for i, (chunk, score) in enumerate(results, 1):
            print(f"\nResult {i}  (Similarity: {score:.3f})")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)


if __name__ == "__main__":
    test_queries()
