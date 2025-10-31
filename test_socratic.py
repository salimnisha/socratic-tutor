# test_socratic.py

"""
Test the Socratic teaching engine
"""

from src.socratic_engine import generate_teaching_question, evaluate_answer

print("=" * 60)
print("TESTING SOCRATIC QUESTION GENERATION")
print("=" * 60)

# ---------TOPIC LIST TESTING-----------------
# # Test different topics
# topics = ["self-supervision", "tokens in language models", "foundation models"]

# for topic in topics:
#     result = generate_teaching_question(
#         topic, pdf_name="chip_huyen_ch_1", difficulty="beginner"
#     )

#     print(f"\n📚 Topic: {topic}")
#     print("\n❓ Question:")
#     print(f"    {result['question']}")
#     print("\n🎯 Teaching goal:")
#     print(f"    {result['teaching_goal']}")
#     print("\n💡 Hint if stuck:")
#     print(f"    {result['hint_if_stuck']}")
# ---------TOPIC LIST TESTING-----------------

# Test a single topic
# Step 1: Generate question
print("\n" + "=" * 60)
print("STEP 1; GENERATE TEACHING QUESTION")
print("=" * 60)

topic = "self-supervision"
question_data = generate_teaching_question(
    topic, pdf_name="chip_huyen_ch_1", difficulty="beginner"
)

print(f"\n📚 Topic: {topic}")
print("\n❓ Question:")
print(f"    {question_data['question']}")

# Simulate different student answers
print("\n\n" + "=" * 60)
print("STEP 2: EVALUATE DIFFERENT ANSWERS")
print("=" * 60)

test_answers = [
    {
        "label": "GOOD ANSWER",
        "answer": "Self-supervision means the model creates its own labels from the data, so you don't need humans to label everything manually.",
    },
    {
        "label": "PARTIAL ANSWER",
        "answer": "It means the model learns by itself without any help",
    },
    {
        "label": "INCORRECT ANSWER",
        "answer": "Self-supervision is when the model supervises other models.",
    },
]

for answer in test_answers:
    print(f"\n{'-' * 60}")
    print(f"Testing: {answer['label']}")
    print(f"{'-' * 60}")

    print("\n💭 Student answer:")
    print(f"    {answer['answer']}")

    evaluation = evaluate_answer(
        question=question_data["question"],
        student_answer=answer["answer"],
        context=question_data["context_used"],
        teaching_goal=question_data["teaching_goal"],
    )

    print("\n📊 Evaluation:")
    print(f"    Correctness: {evaluation['evaluation']['correctness']}")
    print(f"    Strengths: {evaluation['evaluation']['strengths']}")
    print(f"    Gaps: {evaluation['evaluation']['gaps']}")
    if evaluation["evaluation"]["misconceptions"]:
        print(f"    Misconceptions: {evaluation['evaluation']['misconceptions']}")

    print("\n💬 Feedback:")
    print(f"    {evaluation['feedback']}")

    if evaluation["next_question"]:
        print("\n❓ Next question:")
        print(f"    {evaluation['next_question']}")
