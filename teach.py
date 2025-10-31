# teach.py

"""
Interactive Socratic Teaching Session

Usage:
    python teach.py
"""

from src.socratic_engine import teach_topic
import sys


def main():
    """Run interacive teaching session"""
    print("=" * 60)
    print("🎓 SOCRATIC TEACHING SESSION")
    print("=" * 60)

    topics = [
        "self-supervision",
        "tokens",
        "embeddings",
        "foundation models",
        "language models",
    ]

    print("\n Let's learn one of the following topics from your textbook")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")

    user_input = input("\nInput the topic number: ").strip()

    try:
        topic_number = int(user_input)
    except ValueError:
        print("Invalid input. Please enter a valid number")

    if topic_number <= 0 or topic_number > len(topics):
        print("Not a valid selection. Bye!")
        sys.exit(0)

    # Teach the topic
    topic = topics[topic_number - 1]
    session = teach_topic(topic, max_turns=5)

    # Offer to save the transcript
    print("\n" + "=" * 60)
    save = input("\nWould you like to save this session? (y/n) ").strip().lower()

    if save == "y":
        import json
        from datetime import datetime
        from pathlib import Path

        # Create sessions directory
        sessions_dir = Path("sessions")
        sessions_dir.mkdir(exist_ok=True)

        # Save the session
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"session - {topic.replace(' ', '_')}_{timestamp}.json"
        filepath = sessions_dir / filename

        with open(filepath, "w") as f:
            json.dump(session, indent=2, fp=f)

        print(f"✔️ Session saved to {filepath}")

    print("\nThank you for learning with me! 🎓")


if __name__ == "__main__":
    main()
