import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rag import ask_question

print("🤖 AI Document Assistant")
print("Type 'exit' to quit.\n")

# Simulated user session
session_id = input("Enter session id: ").strip()

while True:

    query = input("\nAsk a question: ").strip()

    if query.lower() == "exit":
        print("\n👋 Goodbye!")
        break

    response = ask_question(
        query=query,
        session_id=session_id
    )

    print("\n🧠 Answer:\n")
    print(response["answer"])

    # print("\n📚 Retrieved Sources:\n")

    # for i, source in enumerate(response["sources"]):

    #     print(f"\n--- Source {i+1} ---")

    #     print(f"📄 Page: {source['page']}")

    #     print(f"📝 Preview:\n{source['preview']}")