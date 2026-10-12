# 📚 AI Research & Document Assistant

An AI-powered document question-answering system built using **Retrieval-Augmented Generation (RAG), LangChain, LangGraph, FastAPI, and Streamlit**. It enables users to upload documents and receive context-grounded answers with relevance checking, grounding verification, and source references.

## 🚀 Features

- Upload and process PDF, DOCX, and TXT documents.
- Retrieve relevant document chunks using semantic search.
- Generate context-aware answers using an LLM.
- Verify answer grounding and document relevance.
- Retry retrieval when the context is insufficient.
- Manage document embeddings using ChromaDB.
- Provide REST APIs through FastAPI and an interactive UI through Streamlit.
- Trace and evaluate workflows using LangSmith.

## 🛠️ Tech Stack

Python • LangChain • LangGraph • RAG • ChromaDB • HuggingFace Embeddings • Groq • FastAPI • Streamlit • LangSmith

## 🏗️ Architecture

```text
Document Upload
      ↓
Text Extraction & Chunking
      ↓
Embedding Generation
      ↓
ChromaDB Vector Store
      ↓
LangGraph RAG Workflow
      ↓
Retrieval & Relevance Checking
      ↓
Answer Generation & Grounding Verification
      ↓
Answer + Sources
```

## 📂 Project Structure

```text
AI-Research-&-Document-Intelligence-Assistant/
├── app/
│   ├── api.py
│   ├── config.py
│   ├── graph.py
│   └── rag.py
├── data/
│   ├── documents/
│   └── chroma/
├── evaluation/
│   └── eval_dataset.jsonl
├── scripts/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/vijayn1302/AI-Research-&-Document-Intelligence-Assistant.git
cd AI-Document-Assistant
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file with your required API keys and model settings.

```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=AI Research & Document Intelligence Assistant
```

Never commit real API keys or secrets to GitHub.

## ▶️ Run the Application

Start the FastAPI backend:

```bash
uvicorn app.api:app --reload --port 8000
```

Start the Streamlit frontend in a second terminal:

```bash
streamlit run app/streamlit_app.py
```

- **API Documentation:** http://localhost:8000/docs
- **Streamlit UI:** http://localhost:8501

Adjust the Streamlit filename if your entry point differs.

## 🧪 Evaluation

The evaluation dataset tests answer correctness, retrieval relevance, grounding, and handling of unanswerable questions.

Example:

```json
{"question":"Where should technical issues be reported?","answer":"Technical issues should be reported through the internal support portal.","top_k":4}
```

## 🔮 Future Enhancements

- Conversational memory and chat history.
- Hybrid search and document reranking.
- User authentication and access control.
- Cloud deployment and advanced evaluation metrics.

## 👨‍💻 Author

**Vijay N**  
AI & Machine Learning Enthusiast | Aspiring AI/ML Engineer
