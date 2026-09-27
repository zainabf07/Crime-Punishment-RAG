from rag import ask_rag

questions = [
    "Why did Raskolnikov feel guilty?",
    "Why did Raskolnikov commit the murder?"
]

print("=" * 80)
print("RAG EVALUATION")
print("=" * 80)

for question in questions:

    print("\nQUESTION:")
    print(question)

    answer, documents = ask_rag(question)

    print("\nANSWER:")
    print(answer)

    print("\nRETRIEVED CONTEXT:")
    for i, doc in enumerate(documents, 1):
        print(f"\n--- Context {i} ---")
        print(doc[:1000])

    print("\n" + "=" * 80)