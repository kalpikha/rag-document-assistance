import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rag import ask_question

query = input("Ask a question: ")

response = ask_question(query)

print("\n🧠 Answer:\n")
print(response["answer"])

print("\n📚 Sources:\n")

for source in response["sources"]:
    print(source)
