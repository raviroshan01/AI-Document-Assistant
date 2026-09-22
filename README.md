# 📄 AI Document Assistant

A local AI-powered document question-answering application that allows users to upload a PDF and ask questions about its content.

The project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded document before generating an answer.

## 🚀 Features

- Upload PDF documents
- Extract text from PDF files
- Split documents into smaller chunks
- Generate text embeddings locally
- Store and search document embeddings using Chroma
- Ask natural-language questions about the document
- Generate answers using a local Llama model
- Display relevant PDF page numbers
- Runs completely locally without paid AI APIs

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- PyPDF
- Chroma
- Ollama
- Llama 3.2:1b
- Nomic Embed Text
- RAG (Retrieval-Augmented Generation)

## 🔄 How It Works

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Embeddings
 ↓
Chroma Vector Database
 ↓
User Question
 ↓
Similarity Search
 ↓
Relevant Document Chunks
 ↓
Llama Local LLM
 ↓
Answer
