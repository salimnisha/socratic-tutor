# src/topic_extractor.py

"""Extracts topics and concepts from PDF

Use GPT to analyze the text and list out:
- Main topics
- Concepts
- Short summary

"""

import json
import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

# env and constants
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"
TOPIC_TEMPERATURE = 0.3  # lesser values for more deterministic output


def extract_topics(full_text, pdf_name, max_tokens=100_000):
    """Reads full text, returns all topics and concepts as json

    Args:
        full_text::str
            Complete text from the pdf
        pdf_name::str
            Name of the pdf, for logging
        max_tokens::int
            Maximum tokens to analyze, to stay within token limit

    Returns:
        topic_map::dict
            Topics extracted from the full_text
            {
                "pdf_summary": str,
                "topic_name": {
                    "summary": str,
                    "topics": {
                    "key_points": list,
                    "concepts": list
                    }
                }
            }

    """

    print(f"\n📚 Extracting topics from {pdf_name}...")

    # Check if text within max_length (GPT has token limits)
    text_to_parse = full_text

    encoding = tiktoken.encoding_for_model(MODEL)
    tokens = encoding.encode(text_to_parse)
    if len(tokens) > max_tokens:
        text_to_parse = encoding.decode(tokens[:max_tokens])
        print(f"\n⚠️ PDF token length {len(tokens)} exceeds limit")

    print(f"Parsing {len(text_to_parse)} / {len(full_text)} characters from pdf...")

    # Create the prompt for the model
    messages = [
        {
            "role": "system",
            "content": """ You are a college professor who excels at teaching students complex topics simply and effectively.
                    Your task is to analyze the given text and extract topics and concepts for students to learn.

                    For each topic, provide:
                    1. A short beginner-friendly summary of 2-3 sentences
                    2. A bullet list of 3-5 key points as bullet list
                    3. A bullet list of 3-5 specific concepts to learn to master this topic
                    
                    GUIDELINES:
                    1. Topic names should be short (1-3 words), lowercase with hyphens. e.g., self-supervison, tokens etc.
                    2. Limit to extracting 3-7 main topics. Don't go too granular
                    3. Summary: Explain like teaching a beginner
                    4. Key points: Most important takeaways from the topic
                    5. Concepts: Specific learning objectives or what the student should understand

                    Return JSON with exactly this structure
                    {
                        "pdf_summary": "Short summary of the whole document, 2-3 sentences",
                        "topics": {
                                    "topic_name": {
                                        "summary": "2-3 sentence beginner-friendly explanation",
                                        "key_points": ["key point 1", "key point 2", "key point 3"],
                                        "concepts": ["concept 1", "concept 2", "concept 3"]

                                    }
                                }
                    }

                    EXAMPLE:
                    {
                        "pdf_summary": "Introduction to AI Engineering with an overview of foundation models, how models learn, and some common AI use cases",
                        "topics": {
                                    "tokens": {
                                        "summary": "Tokens are the basic units of text that language models process and understand. Instead of reading words or sentences, models break text into smaller pieces called tokens which may be whole words, parts of words, or even punctuation. The number of tokens determines how much text a model can handle and how much an API call costs.",
                                        "key_points": [
                                            "Tokens are the smallest text units language models read.", 
                                            "A token can be a whole word, subword, or punctuation mark.", 
                                            "Models have a maximum token limit that defines input size."
                                            "API cost and performance depend on the number of tokens used."
                                        ],
                                        "concepts": [
                                            "definition of a token", 
                                            "relationship between tokens and words", 
                                            "how models process tokens sequentially",
                                            "token limits and pricing in APIs"
                                        ]
                                    }
                                }
                    }

            """,
        },
        {
            "role": "user",
            "content": f""" Extract topics and concepts from this text. 
                        {text_to_parse}
                        Return ONLY valid json, nothing else.""",
        },
    ]

    # Calling GPT
    print("\n🤖 Calling GPT to analyze text")

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=TOPIC_TEMPERATURE,
    )

    # Extract the topic map from response
    topic_map = json.loads(response.choices[0].message.content)

    num_topics = len(topic_map.get("topics", {}))
    print(f"\n✓ Extracted {num_topics} topics")
    for i, topic in enumerate(topic_map.get("topics", {}).keys(), 1):
        print(f"{i}. {topic}")

    return topic_map
