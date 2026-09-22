import streamlit as st
from pypdf import PdfReader

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document


# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄"
)

st.title("📄 AI Document Assistant")
st.write("Upload a PDF and ask questions about its content.")


# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file:

    with st.spinner("Processing document..."):

        # Read PDF
        reader = PdfReader(uploaded_file)

        documents = []

        # Extract text page by page
        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": page_number
                        }
                    )
                )

        if not documents:
            st.error("Could not extract text from this PDF.")
            st.stop()


        # -----------------------------
        # Split Document
        # -----------------------------
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)


        # -----------------------------
        # Local Embeddings
        # -----------------------------
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )


        # -----------------------------
        # Vector Database
        # -----------------------------
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )


    st.success("Document processed successfully!")


    # -----------------------------
    # Question
    # -----------------------------
    question = st.text_input(
        "Ask a question about the PDF"
    )


    if question:

        with st.spinner("Finding the answer..."):

            # Retrieve relevant chunks
            relevant_docs = vectorstore.similarity_search(
                question,
                k=4
            )

            # Create context
            context = "\n\n".join(
                doc.page_content
                for doc in relevant_docs
            )


            # -----------------------------
            # Local LLM
            # -----------------------------
            llm = ChatOllama(
                model="llama3.2:1b",
                temperature=0
            )


            # -----------------------------
            # Prompt
            # -----------------------------
            prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the
information provided in the document context.

If the answer is not present in the document,
say: "The answer is not available in the document."

Document Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""


            # Generate answer
            response = llm.invoke(prompt)


        # -----------------------------
        # Display Answer
        # -----------------------------
        st.subheader("Answer")

        st.write(response.content)


        # -----------------------------
        # Show Sources
        # -----------------------------
        st.subheader("Sources")

        pages = sorted(
            set(
                doc.metadata.get("page")
                for doc in relevant_docs
                if doc.metadata.get("page")
            )
        )

        if pages:
            st.write(
                "Relevant pages: "
                + ", ".join(str(page) for page in pages)
            )