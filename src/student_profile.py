# src/student_profile.py

"""Student Profile tracks student topic learning progress
- Which topic learned, which topic weak
- When was the last study session
- Simple persistence to JSON

Phase 1: Only one student profile - "default"
"""

import json
from datetime import datetime
from pathlib import Path


class StudentProfile:
    """Manage the learning progress of a student

    Tracks which concepts are learned and which are weak for each topic
    Saves to student_profiles/ folder as {student_id}.json file
    Default student id is "default"

    """

    # -----------------------------------------------------
    # Initialize
    # -----------------------------------------------------
    def __init__(self, student_id="default", profiles_dir="student_profiles"):
        """Initialize or load student profile

        Args:
            student_id::str
                student identifier, default id = "default"
            profiles_dir::str
                Location to store student profile

        """
        self.student_id = student_id
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)

        self.profile_path = self.profiles_dir / f"{student_id}.json"

        # Load or create the profile json file
        if self.profile_path.exists():
            self.load()
        else:
            self.create_new()

    # -----------------------------------------------------
    # Create a new student profile
    # -----------------------------------------------------
    def create_new(self):
        # Create new profile data and save to disk
        self.data = {
            "student_id": self.student_id,
            "created_at": datetime.now().isoformat(),
            "topics": {},
        }
        self.save()

        print(f"\n✔︎ Created new student profile: {self.student_id}")

    # -----------------------------------------------------
    # Load a student profile when it exists
    # -----------------------------------------------------
    def load(self):
        # Load the profile file
        with open(self.profile_path, "r") as f:
            self.data = json.load(f)
        print(f"✔︎ Loaded profile: {self.student_id}")

    # -----------------------------------------------------
    # Write a student profile to disk as json
    # -----------------------------------------------------
    def save(self):
        # Save file to disk
        with open(self.profile_path, "w") as f:
            json.dump(self.data, fp=f, indent=2)

    # -----------------------------------------------------
    # Retrieve progress of a topic
    # -----------------------------------------------------
    def get_topic_progress(self, topic):
        """Get progress on a specific topic

        Args:
            topic::str
                Topic name

        Returns:
            dict: {
                    "concepts_learned": ["list"],
                    "concepts_weak": ["list"],
                    "last_studied": str (ISO datetime)
                  }
                  or {} if never studied the topic

        """
        return self.data["topics"].get(topic, {})

    # -----------------------------------------------------
    # Update learning progress on a concept that belongs to
    # a topic
    # -----------------------------------------------------
    def update_concept_progress(self, topic, concept, status):
        """Update progress of a concept

        Args:
            topic::str
                Topic name
            concept::str
                Concept within the topic
            status::str
                "learned" or "weak"
        """

        # Create topic if it doesn't exist, else retrieve
        topic_data = self.data["topics"].setdefault(
            topic,
            {
                "concepts_learned": [],
                "concepts_weak": [],
                "last_studied": None,
            },
        )

        # Remove the current concept values to avoid duplicates
        if concept in topic_data["concepts_learned"]:
            topic_data["concepts_learned"].remove(concept)
        if concept in topic_data["concepts_weak"]:
            topic_data["concepts_weak"].remove(concept)

        # Update the new values
        if status == "learned":
            topic_data["concepts_learned"].append(concept)
        elif status == "weak":
            topic_data["concepts_weak"].append(concept)

        # Timestamp
        topic_data["last_studied"] = datetime.now().isoformat()

        self.save()

    # -----------------------------------------------------
    # Display progress on a specific topic or overall
    # -----------------------------------------------------
    def display_progress(self, topic=None, topic_map=None):
        """Display learning progress

        Args:
            topic::str (optional)
                Show progress of a topic or for all topics
            topic_map::list (optional)
                List of all topics and concepts from pdf
        """
        if topic:
            self._display_topic_progress(topic, topic_map)
        else:
            self._display_all_progress(topic_map)

    # -----------------------------------------------------
    # Display progress on topic
    # -----------------------------------------------------
    def _display_topic_progress(self, topic, topic_map=None):
        """Display the learing status of concepts in the topic"""

        # Get the progress data for the topic
        progress = self.get_topic_progress(topic)

        if not progress:
            print(f"📚 {topic}: Not studied yet")
            return

        print(f"\n{'=' * 60}")
        print(f"📊 PROGRESS: {topic}")
        print("=" * 60)

        learned = progress.get("concepts_learned", [])
        weak = progress.get("concepts_weak", [])

        if learned:
            print(f"\n✔︎ Mastered ({len(learned)}):")
            for concept in learned:
                print(f"    • {concept}")

        if weak:
            print(f"\n☒ Need review ({len(weak)}):")
            for concept in weak:
                print(f"    • {concept}")

        if not learned and not weak:
            print("\n📚 No concepts studied yet")

        print("=" * 60)

    # -----------------------------------------------------
    # Display progress on all topics
    # -----------------------------------------------------
    def _display_all_progress(self, topic_map=None):
        """Displays progress for all topics"""
        print(f"\n{'=' * 60}")
        print("📊 YOUR OVERALL PROGRESS")
        print("=" * 60)

        topics = self.data["topics"]

        if not topics:
            print("\n📚 No topics studied yet")
            return

        for i, topic in enumerate(topics, 1):
            print(f"\n{i}. {topic}")
            progress = self.get_topic_progress(topic)
            learned = len(progress.get("concepts_learned", []))
            weak = len(progress.get("concepts_weak", []))

            # Get all concepts for the topic from topic_map
            all_concepts = []
            topic_map_topic = topic_map["topics"].get(topic, {}) if topic_map else {}
            if topic_map_topic:
                all_concepts = topic_map["topics"][topic].get("concepts", [])
            # If empty concept list from topic map, fall back to concepts covered by student
            total = len(all_concepts) if all_concepts else (learned + weak)

            # total = learned + weak

            if total > 0:
                progress_percentage = int((learned / total) * 100)

                num_progress_blocks = progress_percentage // 10
                num_weak_blocks = 10 - num_progress_blocks

                print(
                    f"  [{'█' * num_progress_blocks}"
                    + f"{'░' * num_weak_blocks}]"
                    + f"    {progress_percentage}%"
                    + f" ({learned}/{total} concepts)"
                )

        print(f"\n{'=' * 60}")
