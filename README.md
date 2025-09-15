# ML TutorBot 🤖📚
Your AI Data Science Tutor

ML TutorBot is a multilingual, AI-powered Data Science & Machine Learning tutor. It helps users understand ML/DS concepts, libraries, and techniques in a conversational way, using Retrieval-Augmented Generation (RAG) to provide accurate, contextual answers from curated knowledge sources.

---

## 🗺️ Road Map:
Next steps:
1. Add the initial structure ✅
2. Add the agent workflow 🔁
3. Add the API logic and user interaction
4. Make a simple frontend to improve the usabillity

---

## 🔹 Features
  * Answer ML & Data Science questions in multiple languages

  * Uses RAG to retrieve context from official documentation, tutorials, and open-access books

  * Provides clear explanations, code snippets, and examples

  * Modular agent architecture:

  * Language Detector → detects user query language

  * Retriever Agent → fetches relevant chunks from knowledge base

  * Answering Agent → generates concise answers

  * Translator Agent → ensures responses match user language (optional)

## 📚 Knowledge Sources

  * Official Documentation: scikit-learn, Pandas, NumPy, PyTorch, TensorFlow

  * Open-Access Books: Dive into Deep Learning, fast.ai courses

  * Blogs & Tutorials: Kaggle Learn, Towards Data Science, Analytics Vidhya

  * Wikipedia (ML/DS articles)

  * All documents are chunked and embedded into a vector database for RAG.

## ⚡ Tech Stack

Language Model: Gemini 2.5 Flash / Open-source LLM (no fine-tuning required)

Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Vector Database: FAISS / Chroma / Weaviate

Backend: FastAPI

Interface: Gradio or Streamlit (chat interface)

Optional: WhatsApp / Telegram integration for chatbot deployment

## 🚀 Installation & Setup

1. Clone the repo:
  ```bash
  git clone https://github.com/yourusername/ml-tutorbot.git
  cd ml-tutorbot
  ```
2. Install the dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Run:
  ```bash
  docker build -t tutorbot .
  docker run -p 8000:8000 tutorbot
  ```
