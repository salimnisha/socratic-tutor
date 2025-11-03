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
import time
from openai import OpenAI
from dotenv import load_dotenv
from src.retriever import retrieve_relevant_chunks

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
MODEL = "gpt-4o-mini"
TEACHING_TEMPERATURE = 0.7  # more creative than Q&A, so higher temperature
PRICING = {
    "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000}
}  # Pricing for cost calculation


# -------------------------------------------------
# Calculate cost
# -------------------------------------------------
def calculate_cost(model, input_tokens, output_tokens):
    """Calculate cost of API call"""
    if model not in PRICING:
        return 0.0
    input_cost = input_tokens * PRICING[model]["input"]
    output_cost = output_tokens * PRICING[model]["output"]
    return input_cost + output_cost


# -------------------------------------------------
# Generate socrastic question on the topic
# -------------------------------------------------
def generate_teaching_question(
    topic, pdf_name="chip_huyen_ch_1", difficulty="beginner"
):
    """Generate a socaratic question to teach about a topic.

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
def evaluate_answer(
    question, student_answer, context, teaching_goal, return_metadata=False
):
    """Evaluate student's answers and provide feedback

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
        return_metadata::bool (=False)
            Whether to return metadata dict for logging or not

        Returns:
            result::dict
            {
                "evaluation": {
                    "correctness": "correct" | "partial" | "incorrect",
                    "strenghts": [list of good points],
                    "gaps": [list of missing points],
                    "misconeptions": [list of misunderstandings]
                },
                "feedback": str, # What to tell the student
                "next_question": str # Follow-up question (or None if complete)
            }
            # When return_metadata=True
            result, metadata::dict, dict
    """

    # Start tracking time for logging
    start_time = time.time()

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

    # Prepare metadata for logging
    if return_metadata is True:
        total_time_sec = round(time.time() - start_time, 3)
        tokens_used = response.usage.total_tokens
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost_usd = calculate_cost(MODEL, input_tokens, output_tokens)

        metadata = {
            "total_time_sec": total_time_sec,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
            "context_chars": len(context_text),
            "num_chunks": len(context),
        }

    if return_metadata is True:
        return result, metadata
    else:
        return result


# -------------------------------------------------
# Manage a teaching session
# -------------------------------------------------
def teach_topic(topic, pdf_name="chip_huyen_ch_1", max_turns=5, return_metadata=False):
    """Conduct a complete socratic teaching session on a given topic

    Flow:
    1. Generate an opening question
    2. Get student answer
    3. Evaluate and provide feedback
    4. Ask follow-up question
    5. Repeat till student demonstrates mastery or max_turns reached

    Args:
        topic::str
            Topic to teach
        pdf_name::str
            Name of the textbook
        max_turns::int
            Maximum number of Q&A cycles
        return_metadata::bool (=False)
            Whether to return metadata or not

    Returns:
        session::dict
            Session summary with transcript
            {"topic": str,
             "final_assessment": str or None
            "transcript": [
                    {
                    'evaluation': str
                    'feedback': str
                    }]
            }
        # if return_metada is True
        session, metadata::dict, list[dict, dict,...]
    """
    # A session dict to keep track of the question and answer through all turns
    session = {"topic": topic, "transcript": [], "final_assessment": None}
    metadata = []

    # Generate the first question
    question_data = generate_teaching_question(topic, pdf_name)
    current_question = question_data["question"]
    context = question_data["context_used"]
    teaching_goal = question_data["teaching_goal"]

    # Display
    print(f"\n{'=' * 60}")
    print(f"🎓 SOCRATIC TEACHING SESSION ON {topic}")
    print("=" * 60)

    turn = 0
    while turn < max_turns:
        turn += 1

        # Display
        print(f"\n{'-' * 60}")
        print(f"Turn {turn}/{max_turns}")
        print(f"{'-' * 60}")

        # Ask question to student and get answer
        print(f"🎓 Tutor: {current_question}")
        student_answer = input("\n💭 You: ").strip()

        if not student_answer:
            print("Please provide an answer")
            turn -= 1  # Don't count empty turns
            continue

        # Record in transcript
        session["transcript"].append(
            {"turn": turn, "question": current_question, "answer": student_answer}
        )

        # Evaluate the answer
        result = evaluate_answer(
            current_question,
            student_answer,
            context,
            teaching_goal,
            return_metadata,
        )
        if return_metadata:
            evaluation_data, eval_metadata = result
        else:
            evaluation_data = result

        # Give feedback
        print(f"🎓 Tutor feedback: {evaluation_data['feedback']}")

        # Prepare metadata for logging
        if return_metadata:
            teaching_metadata = {
                "turn": turn,
                "topic": topic,
                "pdf_name": pdf_name,
                "correctness": evaluation_data["evaluation"]["correctness"],
                "num_gaps": len(evaluation_data["evaluation"]["gaps"]),
                "answer_length": len(student_answer),
                "has_followup": bool(evaluation_data.get("next_question")),
                "model": MODEL,
            }
            teaching_metadata.update(eval_metadata)
            metadata.append(teaching_metadata)

        # Record the evaluation and feedback in the latest transcript entry
        session["transcript"][-1]["evaluation"] = evaluation_data["evaluation"]
        session["transcript"][-1]["feedback"] = evaluation_data["feedback"]

        # Check if session is done
        # Done == Answer evaluated as correct and next question not generated (in the prompt to the model to evaluate the answer
        # we have asked to return null for next_question if student has mastered the topic
        if (
            evaluation_data["evaluation"] == "correct"
            and not evaluation_data["next_question"]
        ):
            print(
                f"\n✔️ Excellent! You have demonstrated a solid understanding of {topic}"
            )
            session["final_assessment"] = "mastered"
            break

        # If not mastered, continue with the next question
        if evaluation_data["next_question"]:
            current_question = evaluation_data["next_question"]
        else:
            # If model has not returned a follow-up, generate a new question at the next difficulty level
            question_data = generate_teaching_question(
                topic, pdf_name, difficulty="intermediate"
            )
            current_question = question_data["question"]

        if turn >= max_turns:
            print(f"\n⏰ Session complete! We covered a lot about {topic}")
            session["final_assessment"] = "in progress"

    if return_metadata:
        return session, metadata
    else:
        return session
