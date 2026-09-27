import chromadb

client = chromadb.PersistentClient(
    path="chroma_exp_500"
)

collections = client.list_collections()

print("\nCollections found:")

for collection in collections:
    print(
        collection.name,
        "->",
        collection.count(),
        "documents"
    )