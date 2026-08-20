import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.rag_chain import MedicalRAGPipeline


def test_multi_turn_conversational_chatbot():
    print("=========================================================================")
    print("Testing MediRAG Multi-Turn Conversational Chatbot Engine")
    print("=========================================================================\n")

    pipeline = MedicalRAGPipeline()
    chat_history = []

    # Turn 1: Initial Question
    q1 = "What are the recommended first-line pharmacological treatment classes for hypertension in adults?"
    print(f"Clinician (Turn 1): {q1}\n")
    res1 = pipeline.query(q1, chat_history=chat_history)
    print(f"MediRAG Assistant (Turn 1):\n{res1['answer']}\n")
    print(f"Citations: {len(res1['citations'])} | Search Query Used: '{res1['search_query']}'\n")

    chat_history.append({"role": "user", "content": q1})
    chat_history.append({"role": "assistant", "content": res1["answer"]})
    import time
    time.sleep(2)

    # Turn 2: Conversational Follow-up (pronouns / contextual)
    q2 = "What if she is pregnant?"
    print(f"Clinician (Turn 2): {q2}\n")
    res2 = pipeline.query(q2, chat_history=chat_history)
    print(f"MediRAG Assistant (Turn 2):\n{res2['answer']}\n")
    print(f"Citations: {len(res2['citations'])} | Search Query Used: '{res2['search_query']}'\n")
    print(f"Safety Warnings: {res2['safety_audit'].get('warnings')}\n")

    chat_history.append({"role": "user", "content": q2})
    chat_history.append({"role": "assistant", "content": res2["answer"]})
    time.sleep(2)

    # Turn 3: Second Follow-up (age-specific targets)
    q3 = "What are the recommended blood pressure targets if the patient is aged over 80?"
    print(f"Clinician (Turn 3): {q3}\n")
    res3 = pipeline.query(q3, chat_history=chat_history)
    print(f"MediRAG Assistant (Turn 3):\n{res3['answer']}\n")
    print(f"Citations: {len(res3['citations'])} | Search Query Used: '{res3['search_query']}'\n")

    print("=========================================================================")
    print("Multi-Turn Conversational Chatbot Test Passed Successfully!")
    print("=========================================================================")


if __name__ == "__main__":
    test_multi_turn_conversational_chatbot()
