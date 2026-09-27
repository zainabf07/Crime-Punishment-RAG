import os

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

import chromadb

from dotenv import load_dotenv

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_PATH = "./chroma_exp_500"

COLLECTION_NAME = "crime_500"

TOP_K = 3


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

embedding_model = AutoModel.from_pretrained(
    MODEL_NAME
)

embedding_model.eval()

print("Embedding model loaded.")


# ============================================================
# LOAD CHROMA
# ============================================================

print("Connecting to Chroma...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Chroma loaded: {collection.count()} documents"
)


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = embedding_model(
            **encoded
        )

    token_embeddings = output.last_hidden_state

    attention_mask = encoded[
        "attention_mask"
    ]

    mask = attention_mask.unsqueeze(
        -1
    ).expand(
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

    embedding = (
        sum_embeddings / sum_mask
    )

    embedding = F.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding[
        0
    ].cpu().numpy().tolist()


# ============================================================
# RAG FUNCTION
# ============================================================

def ask_rag(question):

    # --------------------------------------------------------
    # 1. Embed question
    # --------------------------------------------------------

    query_embedding = create_embedding(
        question
    )


    # --------------------------------------------------------
    # 2. Retrieve passages
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=TOP_K
    )

    documents = results[
        "documents"
    ][0]


    # --------------------------------------------------------
    # 3. Build context
    # --------------------------------------------------------

    context = "\n\n".join(
        documents
    )


    # --------------------------------------------------------
    # 4. Prompt Gemini
    # --------------------------------------------------------

    prompt = f"""
You are a helpful RAG assistant for
Crime and Punishment by Fyodor Dostoevsky.

Answer the user's question using the
retrieved passages below.

Do not invent facts.

If the retrieved passages do not contain
enough information, say that the retrieved
passages do not provide enough information.

RETRIEVED PASSAGES:

{context}

USER QUESTION:

{question}

Give a clear and concise answer.
"""


    # --------------------------------------------------------
    # 5. Generate answer
    # --------------------------------------------------------

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )


    # --------------------------------------------------------
    # 6. Return answer and sources
    # --------------------------------------------------------

    return response.text, documents