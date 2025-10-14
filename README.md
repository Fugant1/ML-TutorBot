<p align="center">
 Your AI Data Science Tutor
</p>

<p align="center">
 <img width="326" height="326" alt="ML-TutorBot" src="https://github.com/user-attachments/assets/53e781d7-4a98-41df-9d84-3eff5813382c" />
</p>

ML TutorBot is a multilingual, AI-powered Data Science & Machine Learning tutor. It helps users understand ML/DS concepts, libraries, and techniques in a conversational way, using Retrieval-Augmented Generation (RAG) to provide accurate, contextual answers from curated knowledge sources.

<p align="center">
  <img width="1886" height="893" alt="ML TutorBot UI Screenshot" src="https://github.com/user-attachments/assets/7682c310-06b8-4c69-8a38-8919e4712360" />
</p>
---

## 🗺️ Road Map:
Next steps:
1. Add the initial structure ✅
2. Add the agent workflow ✅
3. Add the API logic and user interaction ✅
4. Make a simple frontend to improve the usabillity ✅
5. Add the language detector and the translator agent 🔁
6. Deploy it in Hostinger or some platform like that

---

## 🔹 Features
  * Answer ML & Data Science questions in multiple languages

  * Uses RAG to retrieve context from official documentation, tutorials, and open-access books

  * Provides clear explanations, code snippets, and examples

  * Modular agent architecture:

  * Language Detector → detects user query language

  * Retriever Agent → fetches relevant chunks from knowledge base

  * Answering Agent → generates concise answers

  * Translator Agent → ensures responses match user language

## 📜 Flow Diagram

```mermaid
flowchart TD

A[User in Gradio Frontend] -->|Sends question/request| B[API - api.py using FastAPI]

B --> C[Agent Workflow]

C -->|Analyzes input intent| D{Select Tool}

D -->|General ML or DS question| E[RAG Pipeline]
D -->|Requires code execution| F[Code Interpreter]

E --> G[Retrieve from Chroma DB]
E --> Z[Scrapp the data from notable source]
Z --> X[Store the scrapped in the Vectorstore]
X --> G[Retriever Agent fetches context from Chroma DB]
G --> J[Provide all the information to the final answering agent]

F --> K[Execute Python logic safely - sandboxed]
K --> L[Return computed result or code output]
L --> J

J --> M[Translator Agent matches user language]
M --> N[API sends response to Frontend]
N -->|Displays explanation or result| O[User sees response in Frontend]
```


## 📁 Repository Structure

```bash
  ML-TutorBot/
├── src/
│   ├── app/
│   │   ├── agent_workflow/        # Handles agent orchestration (Language, Retriever, Answering, Translator)
│   │   ├── api/                   # FastAPI routes and API logic
│   │   ├── core/                  # Core utilities, configs, and constants
│   │   ├── frontend/              # Gradio UI components and design
│   │   ├── rag_pipelines/         # RAG (Retrieval-Augmented Generation) logic and document retrieval flow
│   │   └── __init__.py
│   │
│   ├── data/                      # Preprocessed documents and text datasets for embeddings
│   ├── chroma/                    # Vector database storage (Chroma persistence)
│   ├── tests/                     # Unit and integration tests
│   ├── __init__.py
│   └── main.py                    # Entry point for backend execution
│
├── docker-compose.yml             # Docker multi-service setup (backend, vector DB, etc.)
├── Dockerfile                     # Container definition for ML TutorBot
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── LICENSE                        # License file (if added)
```

## 📚 Knowledge Sources

  * Official Documentation: scikit-learn, Pandas, NumPy, PyTorch, TensorFlow

  * Open-Access Books: Dive into Deep Learning, fast.ai courses

  * Blogs & Tutorials: Kaggle Learn, Towards Data Science, Analytics Vidhya

  * Wikipedia (ML/DS articles)

  * All documents are chunked and embedded into a vector database for RAG.

## ⚡ Tech Stack

Language Model: Gemini 2.5 Flash Lite

Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Vector Database: Chroma

Backend: FastAPI

Interface: Gradio 

## 🚀 Installation & Setup

1. Clone the repo:
  ```bash
  git clone https://github.com/Fugant1/ML-TutorBot.git
  cd ML-TutorBot
  ```
2. Run:
  ```bash
  docker compose build
  docker compose up
  python3 -m app.frontend.ui
  ```
