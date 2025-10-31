# src/socratic_engine.py

"""
Socratic Teaching Engine: Guides learning through questioning.

This module handles:
1. Generating teaching questions based on material
2. Evaluating student's answer
3. Providing targeted feedback
4. Adapting to the student's level
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from src.retriever import retrieve_relevant_chunks

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
MODEL = "gpt-4o-mini"
TEACHING_TEMPERATURE = 0.7  # more creative than Q&A, so higher temperature


# -------------------------------------------------
# Generate socrastic question on the topic
# -------------------------------------------------
def generate_teaching_question(
    topic, pdf_name="chip_huyen_ch_1", difficulty="beginner"
):
    """
    Generate a socaratic question to teach about a topic.

    The question should:
    - Guide discovery (not just test recall)
    - Build on basic concepts
    - Encourage thinking

    Args:
        topic::str
            The topic to teach (e.g., self-supervision)
        pdf_name::str
            Which textbook to use
        difficulty::str
            "beginner", "intermediate", "difficult"

    Returns:
        result::dict
            {
                "question": str,
                "teaching_goal": str,
                "hint_if_stuck": str,
                "context_used": list of chunks
            }
    """

    print(f"\n🎓 Generating teaching question about {topic}")

    # Retrieve relevant material
    context_chunks = retrieve_relevant_chunks(
        query=f"Explain {topic} in detail", pdf_name=pdf_name, top_k=3
    )

    # Extract just the text
    context_text = "\n\n----\n\n".join(chunk for chunk, _ in context_chunks)

    # Create teaching prompt for the model
    messages = [
        {
            "role": "system",
            "content": """ You are a Socratic tutor. Your role is to help students learn by guided questioning, not by directly telling them.
            
                        PRINCIPLES:
                        1. Ask questions that guide discovery
                        2. Start with student's intuition before technical details
                        3. Build from simple to complex
                        4. Encourage thinking, not just recall

                        QUESTION TYPES (use variety):
                        - Intuition: "What do you think X might mean based on the words?"
                        - Connection: "How might X relate to Y that we discussed?"
                        - Analysis: "Why do you think they designed it this way?"
                        - Prediction: "What might happen if we changed X?"

                        Return your response as JSON with this structure:
                        {
                            "question": "Your Socratic question here",
                            "teaching_goal": "What you want the student to understand",
                            "hint_if_stuck": "Gentle hint if student is completely stuck"
                        }
                        """,
        },
        {
            "role": "user",
            "content": f"""Topic to teach: {topic} 
            
                        Difficulty level: {difficulty}

                        Context from textbook: {context_text}

                        Generate a question to help the student discover and understand this concept. The question should guide their thinking, not just test if they memorized the definition.

                        Remember: Return ONLY valid JSON, nothing else.
            """,
        },
    ]

    # Call GPT-4 with JSON mode
    print("🤖 Generating question...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEACHING_TEMPERATURE,
        response_format={"type": "json_object"},  # Force JSON output
    )

    # Parse response
    result = json.loads(response.choices[0].message.content)

    # Add context for later use
    result["context_used"] = [chunk for chunk, _ in context_chunks]

    print(f"✓ Generated question: {result['question'][:100]}...")

    return result


# -------------------------------------------------
# Evaluate the answer given by the student
# -------------------------------------------------
def evaluate_answer(question, student_answer, context, teaching_goal):
    """
    Evaluate student's answers and provide feedback

    Evaluation considers:
    - Correctness (is it right?)
    - Completeness (did it cover key points?)
    - Misconceptions (do they misunderstand something?)

    Feedback should:
    - Acknowledge what's correct
    - Gently guide towards gaps
    - Ask follow-up questions

    Args:
        question::str
            The question asked
        student_answer::str
            Student's response
        context::list
            Relevant chunks retrieved from the textbook
        teaching_goal::str
            What we're trying to teach

        Returns:
            dict: {
                "evaluation": {
                    "correctness": "correct" | "partial" | "incorrect",
                    "strenghts": [list of good points],
                    "gaps": [list of missing points],
                    "misconeptions": [list of misunderstandings]
                },
                "feedback": str, # What to tell the student
                "next_question": str # Follow-up question (or None if complete)
            }
    """

    print("\n🔍 Evaluating answer")

    # Prepare context
    context_text = "\n\n---\n\n".join(context)

    messages = [
        {
            "role": "system",
            "content": """You are a Socratic tutor evaluating a student's answer. 

                    EVALUATION CRITERIA:
                    1. Correctness: Is the core understanding right?
                    2. Completeness: Did they cover the key concepts?
                    3. Misconceptions: Are there any misunderstandings?

                    FEEDBACK PRINCIPLES:
                    1. Always start with what's good (even if wrong, find something!)
                    2. Be encouraging and supportive
                    3. Guide toward gaps with questions, not lectures
                    4. If completely wrong, ask simpler question to build foundation

                    RESPONSE STRUCTURE:
                    Return JSON with:
                    {
                        "evaluation": {
                            "correctness": "correct" | "partial" | "incorrect",
                            "strengths": ["list", "of", "good", "points"],
                            "gaps": ["list", "of", "missing", "concepts"],
                            "misconceptions": ["list", "of", "misunderstandings"]
                        },
                        "feedback": "Your encouraging guiding response to student",
                        "next_question": "Follow-up question to deepen understanding (or null if they have mastered it)"
                    }

                    FEEDBACK EXAMPLES:
                    If CORRECT:
                    "Excellent! You've graspec the key idea that [restate]. Now let's explore [next aspect]"

                    If PARTIAL:
                    "Great thinking! You're right that [correct part]. Let me ask you this: [question about gap]"

                    INCORRECT:
                    "I can see you're thinking hard about this! Let's try a simpler angle: [easier question]"
            """,
        },
        {
            "role": "user",
            "content": f"""Question asked: {question} 
                    Teaching goal" {teaching_goal}
                    Student's answer: {student_answer}
                    Context from textbook: {context_text}

                    Evaluate the student's answer and provide Socratic feedback. Remember: guide, don't tell.

                    Return ONLY valid JSON.
                    """,
        },
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEACHING_TEMPERATURE,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    print(f"✔️ Evaluation: {result['evaluation']['correctness']}")

    return result
