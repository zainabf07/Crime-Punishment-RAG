from rag import ask_rag


question = "Why did Raskolnikov commit the murder?"


print()
print("=" * 80)
print("QUESTION")
print("=" * 80)

print(question)


answer, documents = ask_rag(
    question
)


print()
print("=" * 80)
print("GEMINI ANSWER")
print("=" * 80)

print(answer)


print()
print("=" * 80)
print("RETRIEVED PASSAGES")
print("=" * 80)

for i, document in enumerate(
    documents,
    start=1
):

    print()
    print(f"PASSAGE {i}")
    print("-" * 80)

    print(document[:800])