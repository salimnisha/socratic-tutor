# src/qa_system.py

"""
Question Answering system: Generate answers using RAG

This module handles:
1) Retrieving relevant context
2) Constructing priompts for GPT-4
3) Generating answers
4) Preventing hallucination with strict instructions

"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from src.retriever import retrieve_relevant_chunks

# Load env variables and constants
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

PRICING = {
    "gpt-4o-mini": {
        "input": 0.150 / 1_000_000,  # per token
        "output": 0.600 / 1_000_000,  # per token
    },
    "gpt-4o": {
        "input": 2.50 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
}


def calculate_cost(model, input_tokens, output_tokens):
    """
    Calculate cost of API call

    Args:
        model::str
            Model name
        input_tokens::int
            Number of input tokens
        output_tokens::int
            Number of output tokens

    Returns:
        total_cost::float
            cost in USD
    """
    if model not in PRICING:
        return 0.0  # unknown model

    input_cost = input_tokens * PRICING[model]["input"]
    output_cost = output_tokens * PRICING[model]["output"]
    total_cost = input_cost + output_cost

    return total_cost


def create_qa_prompt(query, context_chunks):
    """Create a prompt for GPT-4 that prevents hallucinations.

    Structure:
    1. System message: Define role and rules
    2. Context: Provide retrieved chunks
    3. Question: User's query

    Args:
        query::str
            Question asked by the user
        context_chunks::list
            List of (chunk, score) tuples

    Returns:
        messages::list
            Messages in OpenAI format
    """

    # Extrract just the text from (text, score) tuples
    chunks_text = [chunk for chunk, score in context_chunks]

    # Join chunks with separators
    context = "\n\n---\n\n".join(chunks_text)

    # Create messages
    messages = [
        {
            "role": "system",
            "content": """ You are a helpful tutor teaching from a specific textbook.
            CRITICAL RULES:
            1. Answer only using the provided context
            2. If the answer is not in the context, say "I cannot find this answer in the material provided"
            3. Quote relevant parts from the context while explaining
            4. Be concise but thorough
            5. Never use outside knowledge - only the context below.""",
        },
        {
            "role": "user",
            "content": f""" Context from textbook: {context}
            Question: {query}
            Please answer solely based on context above.""",
        },
    ]

    return messages


def answer_query(
    query,
    pdf_name="chip_huyen_ch_1",
    top_k=3,
    show_context=False,
    return_metadata=False,
):
    """Answer a question using RAG

    Process:
    1. Retrieve relavant chunks
    2. Create prompt with chunks
    3. Call GPT-4
    4. Return answer

    Args:
        query::str
            The user's question
        pdf_name::str
            Which pdf to search
        top_k::int
            How many chunks to retrieve
        show_context::bool(=False)
            Whether to print retrieved chunks
        return_metadata::bool(=False)
            Whether to return metadata for logging

    Returns:
        answer::str (return_metadata=False)
            Generated answer
        answer, metadata::str, dict (return_metadata=True)
            Generated answer, metadata for logging
    """

    print(f"\n{'=' * 60}")
    print(f"Question: {query}")
    print("=" * 60)

    # Step 1: Retrieve relevant chunks
    retrieval_start = time.time()
    context_chunks = retrieve_relevant_chunks(query, pdf_name, top_k=3)

    # For logging
    retrieval_time = (time.time() - retrieval_start) * 1000  # convert to milliseconds
    similarity_scores = [score for _, score in context_chunks]

    # Optionally show what was retrieved
    if show_context:
        print("\n📚 Retrieved context:")
        for i, (chunk, score) in enumerate(context_chunks, 1):
            print(f"\n--- Chunk {i} (Score: {score:.3f}) ---")
            print(chunk[:200] + "...")

    # Step 2: Create prompt
    messages = create_qa_prompt(query, context_chunks)

    # Step 3: Call GPT-4
    print("\n🤖 Generating answer...")
    temperature = 0.3
    max_tokens = 500
    qa_start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,  # Lower - more focused, less creative
        max_tokens=max_tokens,  # Limit response length
    )

    # Calculate time taken to answer (for logging)
    qa_time = (time.time() - qa_start) * 1000  # convert to milliseconds

    # Step 4: Extract answer
    answer = response.choices[0].message.content

    # Get token usage from response (for logging)
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # Calculate token cost (for logging)
    total_cost = calculate_cost(MODEL, input_tokens, output_tokens)

    # Construct metadata to return
    metadata = {
        "query": query,
        "pdf_name": pdf_name,
        "top_k": top_k,
        "model": MODEL,
        "temperature": temperature,
        "max_tokens_limit": max_tokens,
        "avg_similarity": round(sum(similarity_scores) / len(similarity_scores), 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(total_cost, 6),
        "retrieval_time_ms": round(retrieval_time, 6),
        "qa_time_ms": round(qa_time, 6),
        "total_time_sec": round((retrieval_time + qa_time) / 1000, 4),
        "notes": f"{total_tokens} tokens, avg_similarity: {round(sum(similarity_scores) / len(similarity_scores), 3)}, time_secs: {round((retrieval_time + qa_time) / 1000, 2)}",
    }

    print("\n💡 Answer:")
    print(answer)
    print("\n📊 Stats")
    print(f"     Tokens: {input_tokens} in, {output_tokens} out")
    print(f"     Cost: {total_cost:.6f}")
    print(f"     Time: {metadata['total_time_sec']}")
    print(f"     Avg similarity: {metadata['avg_similarity']}")
    print("=" * 60)

    if return_metadata:
        return answer, metadata
    else:
        return answer
