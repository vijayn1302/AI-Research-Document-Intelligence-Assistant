import streamlit as st
import requests
from pathlib import Path

st.set_page_config(page_title = "AI Document Assistant", page_icon = "📚", layout = "wide")

st.title("📚 AI Document Assistant")
st.caption("RAG + LangChain + LangGraph + FastAPI")

API_URL = st.sidebar.text_input("API URL", "http://localhost:8000").rstrip("/")

SUPPORTED = [".pdf", ".docx", ".txt"]

# ---------- Upload Documents ----------

with st.sidebar:
    st.header("📤 Upload Documents")

    uploaded = st.file_uploader("Choose a PDF, DOCX, or TXT file", type = ["pdf", "docx", "txt"])

    if st.button("Upload & Index", use_container_width=True):
        if uploaded is not None:
            try:
                with st.spinner("Uploading and indexing document..."):
                    response = requests.post(
                        f"{API_URL}/upload",
                        files = {
                            "file": (
                                uploaded.name,
                                uploaded.getvalue(),
                            )
                        },
                        timeout = 120,
                    )

                if response.ok:
                    st.success(
                        response.json().get(
                            "message",
                            "Document uploaded and indexed successfully."
                        )
                    )
                    st.rerun()
                else:
                    try:
                        detail = response.json().get(
                            "detail", response.text
                        )
                    except ValueError:
                        detail = response.text

                    st.error(
                        f"Upload failed ({response.status_code}): {detail}"
                    )

            except requests.exceptions.RequestException as e:
                st.error(f"Upload failed: {e}")
        else:
            st.warning("Please select a document.")

# ---------- Ask Questions ----------

question = st.text_area(
    "Ask a question about your documents",
    height = 100,
)

if st.button("🔎 Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching documents and generating an answer..."):
                response = requests.post(
                    f"{API_URL}/api/ask",
                    json = {"question": question.strip(), "top_k": 4},
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()

            st.subheader("💡 Answer")
            st.write(data.get("answer", "No answer returned."))

            col1, col2 = st.columns(2)
            col1.metric(
                "Grounding Score",
                f'{data.get("grounding_score", 0):.2f}'
            )
            col2.metric(
                "Grounded",
                "Yes" if data.get("grounded", False) else "No"
            )

            if data.get("grounding_reason"):
                st.info(data["grounding_reason"])

            st.subheader("📌 Sources")
            sources = data.get("sources", [])

            if sources:
                for i, source in enumerate(sources, 1):
                    label = source.get("source", "Document")
                    page = source.get("page")

                    if page is not None:
                        label += f" — Page {page}"

                    with st.expander(f"{i}. {label}"):
                        st.write(source.get("snippet", "No preview available."))
            else:
                st.info("No sources available.")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI. Check that the API is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
        except (ValueError, KeyError, TypeError) as e:
            st.error(f"Unable to process the API response: {e}")


# ---------- Delete Confirmation ----------

@st.dialog("🗑️ Delete Document")
def confirm_delete(filename):
    st.warning(f"Are you sure you want to delete **{filename}**?")
    st.write("This action removes the document from the knowledge base.")

    col1, col2 = st.columns(2)

    if col1.button("Cancel", use_container_width = True):
        st.rerun()

    if col2.button("Delete", type = "primary", use_container_width = True):
        try:
            with st.spinner("Deleting document and updating index..."):
                response = requests.delete(
                    f"{API_URL}/documents/{filename}",
                    timeout=120,
                )
                response.raise_for_status()

            st.success(response.json().get(
                "message", "Document deleted successfully."
            ))
            st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"Unable to delete document: {e}")


# ---------- Document Management ----------

with st.sidebar:
    st.divider()
    st.header("📂 Document Management")

    documents_dir = Path(__file__).resolve().parent / "data" / "documents"

    documents = (
        sorted(
            file.name
            for file in documents_dir.iterdir()
            if file.is_file() and file.suffix.lower() in SUPPORTED
        )
        if documents_dir.exists()
        else []
    )

    if documents:
        selected = st.selectbox("Select document", documents)

        if st.button("🗑️ Delete Document", use_container_width = True):
            confirm_delete(selected)
    else:
        st.info("No documents uploaded yet.")