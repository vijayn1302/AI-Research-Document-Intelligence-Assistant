from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (PyPDFLoader, TextLoader, Docx2txtLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import (DOCS_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL)

SUPPORTED = {".pdf", ".txt", ".docx"}
_embeddings = None

# ---------- Load Documents ----------

def load_documents(directory: Path = DOCS_DIR) -> list[Document]:
    documents = []

    if not directory.exists():
        return documents

    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding = "utf-8", autodetect_encoding = True)

        for doc in loader.load():
            doc.metadata.update({
                "source": path.name,
                "path": str(path)
            })
            documents.append(doc)

    return documents

# ---------- Split Documents ----------

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)

# ---------- Load Embedding Model ----------

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name = EMBEDDING_MODEL
        )

    return _embeddings

# ---------- Connect to Chroma ----------

def get_vectorstore() -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name = "documents",
        persist_directory = str(CHROMA_DIR),
        embedding_function = get_embeddings()
    )

# ---------- Rebuild Vector Index ----------

def rebuild_index():
    print(f"📂 Loading documents from: {DOCS_DIR}")

    documents = load_documents()

    print(f"📄 Documents loaded: {len(documents)}")

    if not documents:
        raise ValueError(
            f"No supported documents found in {DOCS_DIR}"
        )

    chunks = split_documents(documents)

    print(f"✂️ Text chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No text chunks were created.")

    embeddings = get_embeddings()

    # Connect to the existing collection
    vectorstore = get_vectorstore()

    # Remove old indexed content before rebuilding
    vectorstore.delete_collection()

    # Create a fresh index from the current documents
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        collection_name = "documents",
        persist_directory = str(CHROMA_DIR)
    )

    print(
        f"✅ Indexed chunks: "
        f"{vectorstore._collection.count()}"
    )

    return {
        "vectorstore": vectorstore,
        "documents": len(documents),
        "chunks": vectorstore._collection.count()
    }

# ---------- Retrieve Relevant Documents ----------
def retrieve(
    question: str,
    top_k: int = 4,
) -> list[tuple[Document, float]]:
    store = get_vectorstore()

    return store.similarity_search_with_score(question, k = top_k)