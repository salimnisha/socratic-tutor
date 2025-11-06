# test_profile.py

"""Test student profile creation and updation"""

from src.student_profile import StudentProfile
from src.vector_store import VectorStore

print("\n" + "=" * 60)
print("TESTING STUDENT PROFILE")
print("=" * 60)

# Test 1: Create or load default profile (should be empty the first time)
print("\n[Test 1]: Creating or loading profile")
profile = StudentProfile()

# Test 2: Show initial progress
print("\n[Test 2]: Initial progress")
store = VectorStore()
topic_map = store.load_topics(pdf_name="chip_huyen_ch_1")
profile.display_progress(topic_map=topic_map)

# Test 3: Update some concepts
print("\n[Test 3]: Learning some concepts...")
profile.update_concept_progress("self-supervision", "definition", "learned")
profile.update_concept_progress(
    "self-supervision", "comparison with supervised", "weak"
)
profile.update_concept_progress("tokens", "what tokens are", "learned")
print("✔︎ Updated concepts")

# Test 4: Show progress after update
print("\n[Test 4]: Progress after learning:")
profile.display_progress(topic_map=topic_map)

# Test 5: Show specific topic progress
print("\n[Test 5]: Progress on self-supervision:")
profile.display_progress("self-supervision")

# Test 6: Check if file exists
print("\n[Test 6]: Check saved file:")
print(f"File: {profile.profile_path}")
print(f"Exists: {profile.profile_path.exists()}")

print(f"\n{'=' * 60}")
print("ALL TESTS COMPLETE")
print("=" * 60)
