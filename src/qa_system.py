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
from openai import OpenAI
from dotenv import load_dotenv
from src.retriever import retrieve_relevant_chunks

# Load env variables and constants
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


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


def answer_query(query, pdf_name="chip_huyen_ch_1", top_k=3, show_context=False):
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

    Returns:
        answer::str
            Generated answer
    """

    print(f"\n{'=' * 60}")
    print(f"Question: {query}")
    print("=" * 60)

    # Step 1: Retrieve relevant chunks
    context_chunks = retrieve_relevant_chunks(query, pdf_name, top_k=3)

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
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,  # Lower - more focused, less creative
        max_tokens=500,  # Limit response length
    )

    # Step 4: Extract answer
    answer = response.choices[0].message.content

    print("\n💡 Answer:")
    print(answer)
    print("=" * 60)

    return answer
