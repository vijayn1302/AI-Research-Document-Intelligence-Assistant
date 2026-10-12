from app.rag import rebuild_index

if __name__ == "__main__":
    result = rebuild_index()
    print(f"Indexed {result['documents']} documents into {result['chunks']} chunks.")
