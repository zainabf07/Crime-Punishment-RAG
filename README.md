# 📚 Crime & Punishment — Retrieval-Augmented Generation (RAG)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-5A67D8)](https://www.trychroma.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface\&logoColor=black)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini%20LLM-4285F4?logo=google\&logoColor=white)](https://ai.google.dev/)
[![RAG](https://img.shields.io/badge/Architecture-RAG-purple)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
[![License](https://img.shields.io/badge/License-Educational-green)](#license)

> **A practical Retrieval-Augmented Generation (RAG) application for question answering over Fyodor Dostoevsky's *Crime and Punishment*.**

The project combines **semantic search, vector databases, embeddings, and Large Language Models (LLMs)** to retrieve relevant passages from the book and use them as context for generating grounded answers.

---

## 📌 Overview

Large Language Models can generate fluent answers, but they may sometimes provide information that is not directly supported by a particular source.

This project demonstrates how **Retrieval-Augmented Generation (RAG)** can address this problem.

Instead of relying only on the language model:

```text
Question → LLM → Answer
```

the system first retrieves relevant information from the source material:

```text
Question
   ↓
Query Embedding
   ↓
Semantic Search
   ↓
ChromaDB
   ↓
Relevant Passages
   ↓
Gemini
   ↓
Grounded Answer
```

The application also displays the retrieved passages so that users can inspect the evidence used by the RAG system.

---

# 🎯 Objectives

The project aims to demonstrate how to:

* Build a complete RAG pipeline.
* Generate semantic embeddings for user queries.
* Store and retrieve document embeddings using ChromaDB.
* Perform semantic similarity search.
* Provide retrieved context to an LLM.
* Generate source-grounded answers.
* Build a simple RAG interface using Streamlit.
* Test retrieval independently from generation.
* Perform basic RAG evaluation.

---

# 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with Large Language Models.

A traditional LLM workflow looks like:

```text
User Question
      ↓
     LLM
      ↓
   Answer
```

A RAG workflow looks like:

```text
                         ┌───────────────┐
                         │ User Question │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   Embedding   │
                         │     Model     │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   ChromaDB    │
                         │ Vector Search │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Top-K Relevant│
                         │    Passages   │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Context +     │
                         │   Question    │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Gemini LLM    │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Grounded      │
                         │    Answer     │
                         └───────────────┘
```

This approach allows the LLM to use retrieved evidence instead of relying entirely on its internal knowledge.

---

# 🛠️ Technology Stack

| Component            | Technology                               |
| -------------------- | ---------------------------------------- |
| Programming Language | Python                                   |
| LLM                  | Google Gemini                            |
| Embedding Model      | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database      | ChromaDB                                 |
| Deep Learning        | PyTorch                                  |
| NLP Framework        | Hugging Face Transformers                |
| Web Interface        | Streamlit                                |
| Configuration        | python-dotenv                            |

---

# 📂 Project Structure

```text
Week8_Day3_RAG/
│
├── app.py
├── rag.py
│
├── evaluate_rag.py
├── test_rag.py
├── test_retrieval.py
├── test_chroma.py
│
├── evaluation_report.txt
│
├── chroma_exp_500/
│   └── ChromaDB vector database
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 📄 File Descriptions

## `app.py`

The main **Streamlit web application**.

It provides:

* Question input
* RAG query execution
* Generated answer
* Retrieved source passages
* Expandable context sections

Run with:

```bash
streamlit run app.py
```

---

## `rag.py`

Contains the core RAG implementation.

The main function is:

```python
ask_rag(question)
```

The function performs:

1. Query embedding
2. ChromaDB retrieval
3. Context construction
4. Gemini prompt generation
5. Answer generation
6. Retrieval result return

---

## `test_retrieval.py`

Tests the retrieval component independently.

It allows you to verify:

* Query embedding
* ChromaDB search
* Top-K retrieval
* Retrieval distances

This is useful for diagnosing retrieval quality before testing the LLM.

---

## `test_chroma.py`

Checks the ChromaDB database.

It verifies:

* Available collections
* Collection names
* Number of documents

Expected collection:

```text
crime_500
```

---

## `test_rag.py`

Tests the complete RAG pipeline from the command line.

Example question:

```text
Why did Raskolnikov commit the murder?
```

The script displays:

* User question
* Generated answer
* Retrieved passages

---

## `evaluate_rag.py`

Runs predefined evaluation questions against the RAG pipeline.

Example:

```text
Why did Raskolnikov feel guilty?

Why did Raskolnikov commit the murder?
```

The results can be used for qualitative assessment of retrieval and generation.

---

## `evaluation_report.txt`

Contains the current evaluation observations and results.

---

# 🔧 Configuration

The main configuration includes:

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_PATH = "./chroma_exp_500"

COLLECTION_NAME = "crime_500"

TOP_K = 3
```

### Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert questions into numerical vector representations.

### Vector Database

The vector database is:

```text
ChromaDB
```

with the persistent database stored in:

```text
./chroma_exp_500
```

### Collection

```text
crime_500
```

### Retrieval

The system retrieves:

```text
TOP_K = 3
```

relevant passages for each question.

---

# 🔑 API Key Setup

The application requires a Google Gemini API key.

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> ⚠️ **Never commit your real API key to GitHub.**

Add the following to `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
.venv/
venv/
```

For sharing the project, you can create:

```text
.env.example
```

with:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Week8_Day3_RAG
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install torch transformers chromadb python-dotenv google-genai streamlit
```

---

## 4. Configure Gemini API

Create:

```text
.env
```

and add:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 5. Verify ChromaDB

Run:

```bash
python test_chroma.py
```

This verifies that the vector database and collection are accessible.

---

## 6. Test Retrieval

Run:

```bash
python test_retrieval.py
```

This tests semantic retrieval independently.

---

## 7. Test the Complete RAG Pipeline

Run:

```bash
python test_rag.py
```

---

## 8. Launch the Web Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 💻 Using the Application

After launching Streamlit, enter a question in the input field.

For example:

```text
Why did Raskolnikov feel guilty?
```

The system performs:

```text
1. Convert question to embedding
          ↓
2. Search ChromaDB
          ↓
3. Retrieve top 3 passages
          ↓
4. Build context
          ↓
5. Send context + question to Gemini
          ↓
6. Generate answer
          ↓
7. Display answer + retrieved passages
```

---

# 💡 Example Questions

Try questions such as:

```text
Why did Raskolnikov feel guilty?

Why did Raskolnikov commit the murder?

What motivated Raskolnikov?

How did guilt affect Raskolnikov?

What role did Sonia play in Raskolnikov's life?

What philosophical ideas influenced Raskolnikov?

How did Raskolnikov's conscience affect his actions?
```

---

# 🧪 Testing

## Test 1 — ChromaDB

```bash
python test_chroma.py
```

Purpose:

```text
Verify vector database
        ↓
Verify collection
        ↓
Verify document count
```

---

## Test 2 — Retrieval

```bash
python test_retrieval.py
```

Purpose:

```text
Question
   ↓
Embedding
   ↓
Vector Search
   ↓
Top-K Passages
```

---

## Test 3 — Complete RAG

```bash
python test_rag.py
```

Purpose:

```text
Question
   ↓
Retrieval
   ↓
Context
   ↓
Gemini
   ↓
Answer
```

---

## Test 4 — Evaluation

```bash
python evaluate_rag.py
```

Purpose:

```text
Evaluation Questions
        ↓
       RAG
        ↓
Generated Answers
        ↓
Retrieved Context
        ↓
Qualitative Evaluation
```

---

# 📊 Evaluation

The current evaluation uses sample questions such as:

### Question 1

> Why did Raskolnikov feel guilty?

The retrieved passages contain information related to:

* Raskolnikov's conscience
* Shame
* Psychological conflict
* His reaction after the murder

### Question 2

> Why did Raskolnikov commit the murder?

The retrieved context contains information related to:

* His motivations
* His philosophical ideas
* His reasoning surrounding the murder

The current evaluation is primarily **qualitative** and should be expanded with a larger benchmark for rigorous performance measurement.

---

# 📈 Recommended RAG Evaluation Metrics

For a more advanced evaluation, the following metrics can be introduced:

| Metric            | Purpose                                            |
| ----------------- | -------------------------------------------------- |
| Recall@K          | Measures whether relevant passages are retrieved   |
| Precision@K       | Measures retrieval relevance                       |
| MRR               | Measures ranking quality                           |
| Context Relevance | Measures usefulness of retrieved context           |
| Faithfulness      | Measures whether answers are supported by context  |
| Answer Relevance  | Measures whether the answer addresses the question |

---

# 🏗️ System Architecture

```text
                         ┌──────────────────┐
                         │      User        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Question     │
                         └────────┬─────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Sentence Transformer   │
                     │ all-MiniLM-L6-v2       │
                     └────────────┬───────────┘
                                  │
                           Query Embedding
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │       ChromaDB         │
                     │     Vector Search      │
                     └────────────┬───────────┘
                                  │
                              Top-K = 3
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Retrieved Passages     │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │   Context Construction │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │     Google Gemini      │
                     │       LLM              │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │   Grounded Answer      │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Supporting Passages    │
                     └────────────────────────┘
```

---

# 🔄 RAG Workflow

The complete workflow can be summarized as:

```text
             USER QUESTION
                    │
                    ▼
           ┌────────────────┐
           │ Query Embedding│
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │   ChromaDB     │
           │ Semantic Search│
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │ Top 3 Passages │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │ Context + Query│
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │ Gemini LLM     │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │ Final Answer   │
           └────────────────┘
```

---

# 🧩 Key Concepts Demonstrated

This project demonstrates several important concepts in modern AI:

### 1. Embeddings

Text is converted into numerical vectors that represent semantic meaning.

### 2. Semantic Search

Instead of searching for exact keywords, the system searches for passages that are semantically similar to the question.

### 3. Vector Database

ChromaDB stores embeddings and allows efficient similarity search.

### 4. Retrieval-Augmented Generation

Relevant source material is retrieved before generating the answer.

### 5. Prompt Engineering

The LLM is instructed to use the retrieved context and avoid unsupported information.

### 6. LLM Integration

The project demonstrates how an external LLM API can be integrated into a Python application.

---

# ⚠️ Limitations

The current version is a learning-focused RAG prototype.

Important limitations include:

* Small evaluation dataset.
* Limited number of evaluation questions.
* Retrieval uses only Top-K = 3.
* No automated Recall@K or MRR evaluation.
* No semantic re-ranking.
* No hybrid keyword + vector search.
* No conversation memory.
* Source metadata is limited.
* Generation depends on the quality of retrieved passages.
* Gemini API access is required.

---

# 🚀 Future Improvements

## 🔹 1. Better Chunking

Experiment with:

* Chunk size
* Chunk overlap
* Sentence-based chunking
* Chapter-aware chunking

---

## 🔹 2. Better Embedding Models

Compare:

```text
all-MiniLM-L6-v2
        vs.
larger sentence-transformer models
```

and evaluate retrieval performance.

---

## 🔹 3. Re-Ranking

Add a re-ranker after the initial vector search:

```text
Question
   ↓
Vector Search
   ↓
Top 10–20 Candidates
   ↓
Re-Ranker
   ↓
Top 3 Context Passages
```

---

## 🔹 4. Hybrid Search

Combine:

```text
Keyword Search
      +
Semantic Search
      ↓
Better Retrieval
```

---

## 🔹 5. Source Citations

Store metadata such as:

```text
Chapter
Page
Section
Chunk ID
Book
```

and display citations with generated answers.

---

## 🔹 6. Automated Evaluation

Create a benchmark dataset containing:

```text
100+ Questions
       ↓
Expected Answers
       ↓
Relevant Passages
       ↓
Automated Evaluation
```

---

## 🔹 7. Conversation Memory

Support follow-up questions:

```text
User:
Why did Raskolnikov feel guilty?

User:
How did this affect his relationship with Sonia?

User:
What happened afterward?
```

---

## 🔹 8. Advanced RAG Architecture

The project could eventually evolve into:

```text
                ┌───────────────┐
                │ User Question │
                └───────┬───────┘
                        │
                        ▼
                Query Rewriting
                        │
                        ▼
                 Hybrid Retrieval
                 /             \
                /               \
       Vector Search         Keyword Search
                \               /
                 \             /
                  ▼           ▼
                    Re-Ranking
                        │
                        ▼
                 Context Filtering
                        │
                        ▼
                   Gemini LLM
                        │
                        ▼
              Citation Generation
                        │
                        ▼
                 Final Answer
```

---

# 📚 Learning Outcomes

After completing this project, students should be able to:

* Explain Retrieval-Augmented Generation.
* Explain semantic embeddings.
* Understand vector similarity search.
* Work with ChromaDB.
* Integrate an LLM into a Python application.
* Construct prompts using retrieved context.
* Build a Streamlit AI application.
* Test retrieval separately from generation.
* Perform basic RAG evaluation.
* Identify limitations of basic RAG systems.
* Propose improvements to a RAG architecture.

---

# 🔐 Security Notes

Never commit sensitive information such as:

```text
API Keys
Passwords
Access Tokens
Private Credentials
```

Make sure `.env` is included in `.gitignore`.

Before pushing to GitHub, check:

```bash
git status
```

and make sure `.env` is **not** included in the files being committed.

---

# 📦 Suggested `requirements.txt`

```text
torch
transformers
chromadb
python-dotenv
google-genai
streamlit
```

For reproducible deployments, package versions can later be pinned after confirming the working environment.

---

# 📄 Suggested `.gitignore`

```gitignore
# Environment variables
.env

# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
venv/
.venv/
env/

# IDE
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

---

# 📝 Example `.env.example`

```env
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here
```

Rename/copy this file to `.env` and replace the placeholder with your actual API key.

---

# 📌 Project Status

**Status:** 🟢 Working Educational Prototype

The current implementation demonstrates the complete RAG pipeline:

```text
Question
   ↓
Embedding
   ↓
Vector Retrieval
   ↓
Context Construction
   ↓
Gemini Generation
   ↓
Answer + Retrieved Evidence
```

The project can be extended into a production-quality document question-answering system by improving retrieval, evaluation, citation handling, security, and user experience.

---

# 🎓 Educational Context

This project is designed as a practical demonstration of **Retrieval-Augmented Generation** and can be used to understand the interaction between:

```text
NLP
 +
Embeddings
 +
Vector Databases
 +
Information Retrieval
 +
Large Language Models
 +
Generative AI
```

It provides a complete example of moving from a simple LLM application toward a grounded, retrieval-based AI system.

---

# 📜 License

This repository is intended primarily for **educational and academic purposes**.

The project uses literary source material for demonstration. Users should ensure that any redistribution or deployment of source text complies with applicable copyright and licensing requirements.

---

# 🙌 Acknowledgments

Built using:

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [Hugging Face Transformers](https://huggingface.co/)
* [ChromaDB](https://www.trychroma.com/)
* [Google Gemini](https://ai.google.dev/)
* [Streamlit](https://streamlit.io/)

---

## ⭐ If You Found This Useful

If this project helped you understand RAG, consider giving the repository a ⭐ on GitHub.

---

**Built for learning, experimentation, and practical understanding of Retrieval-Augmented Generation.**
