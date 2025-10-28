# ask.py

"""
Interactive Q&A interface

Usage:
    python ask.py

Then type quetions and get answers
"""

from src.qa_system import answer_query
import sys


def main():
    """
    Interactive question answering loop
    """

    print("=" * 60)
    print("📚 TEXTBOOK Q&A SYSTEM")
    print("=" * 60)
    print("\nCurrently loaded: Chuip Huyen - AI Engineering, Chapter 1")
    print("\nAsk questions about the material!")
    print("Type quit or exit to stop")

    pdf_name = "chip_huyen_ch_1"

    while True:
        # Get question from user
        try:
            question = input("❓ Your question: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)

        # Check for exit commands
        if question.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            sys.exit(0)

        # Check for empty questions
        if not question:
            continue

        # Pass the question to generate answer
        # Set show_context to false for cleaner output
        try:
            answer_query(question, pdf_name, top_k=3, show_context=False)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure you have run process_pdf.py first")

        print()  # Extra line for readability


if __name__ == "__main__":
    main()
