# test_socratic.py

"""
Test the Socratic teaching engine
"""

from src.socratic_engine import generate_teaching_question

print("=" * 60)
print("TESTING SOCRATIC QUESTION GENERATION")
print("=" * 60)

# Test different topics
topics = ["self-supervision", "tokens in language models", "foundation models"]

for topic in topics:
    result = generate_teaching_question(
        topic, pdf_name="chip_huyen_ch_1", difficulty="beginner"
    )

    print(f"\n📚 Topic: {topic}")
    print("\n❓ Question:")
    print(f"    {result['question']}")
    print("\n🎯 Teaching goal:")
    print(f"    {result['teaching_goal']}")
    print("\n💡 Hint if stuck:")
    print(f"    {result['hint_if_stuck']}")
