import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_PATH = "./chroma_exp_500"
COLLECTION_NAME = "crime_500"

TOP_K = 3


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading embedding model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Embedding model loaded successfully.")


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def create_embedding(text):

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model(**encoded)

    token_embeddings = output.last_hidden_state

    attention_mask = encoded["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    sum_embeddings = torch.sum(
        token_embeddings * mask,
        dim=1
    )

    sum_mask = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    embedding = sum_embeddings / sum_mask

    # Normalize for cosine similarity
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding[0].cpu().numpy().tolist()


# ============================================================
# CONNECT TO CHROMA
# ============================================================

print("Connecting to Chroma...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Collection:", COLLECTION_NAME)
print("Documents:", collection.count())


# ============================================================
# TEST QUESTION
# ============================================================

question = "Why did Raskolnikov feel guilty?"

print()
print("=" * 80)
print("QUESTION:", question)
print("=" * 80)


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

print("Creating query embedding...")

query_embedding = create_embedding(
    question
)

print(
    "Embedding dimensions:",
    len(query_embedding)
)


# ============================================================
# SEARCH CHROMA
# ============================================================

print("Searching Chroma...")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

documents = results["documents"][0]

distances = results["distances"][0]

for i, (document, distance) in enumerate(
    zip(documents, distances),
    start=1
):

    print()
    print("-" * 80)
    print(f"RESULT {i}")
    print("-" * 80)

    print("Distance:", round(distance, 4))

    print()
    print(document[:1200])