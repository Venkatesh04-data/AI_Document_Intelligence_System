# AI Document Intelligence System

An AI-powered application that allows users to upload PDF documents and ask questions about their contents.
The system retrieves relevant sections of the document using semantic search and generates answers using a large language model.

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to provide accurate answers grounded in the uploaded documents.

---

## Features

* Upload one or multiple PDF documents
* Automatically extract and process document text
* Semantic search over document content
* AI-generated answers based on document context
* Source context display for transparency
* Interactive chat-style interface
* Fast vector similarity search

---

## System Architecture

The application follows a **Retrieval-Augmented Generation (RAG)** workflow:

1. **PDF Upload**
   Users upload one or more PDF documents.

2. **Text Extraction**
   The system extracts text from PDFs.

3. **Text Chunking**
   Large documents are split into smaller chunks for better processing.

4. **Embedding Generation**
   Each chunk is converted into a vector embedding using a transformer model.

5. **Vector Storage**
   Embeddings are stored in a FAISS vector index.

6. **User Query Processing**
   The user question is converted into an embedding.

7. **Semantic Retrieval**
   FAISS finds the most relevant document chunks.

8. **Answer Generation**
   A large language model generates an answer using the retrieved context.

---

## Tech Stack

Frontend / UI

* Streamlit

Backend / Processing

* Python

Document Processing

* pypdf

Embedding Model

* SentenceTransformers (all-MiniLM-L6-v2)

Vector Database

* FAISS

Large Language Model

* Cohere (Command model)

---

## Project Structure

```
AI-Document-Intelligence-System
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/ai-document-intelligence-system.git
cd ai-document-intelligence-system
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application:

```
streamlit run app.py
```

Then open the local server in your browser:

```
http://localhost:8501
```

---

## How to Use

1. Enter your Cohere API key.
2. Upload one or more PDF documents.
3. Ask a question about the document.
4. The system retrieves relevant content and generates an answer.

Example:

```
Question:
What technical skills does the candidate have?

Answer:
The candidate has skills in Python, SQL, APIs, Flask, Streamlit, and DBMS.
```

---

## Key AI Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Semantic Embeddings
* Vector Similarity Search
* Transformer-based Language Models
* Document Question Answering

---

## Future Improvements

* PDF viewer with highlighted answer sections
* Support for more document formats (DOCX, TXT)
* Chat memory across multiple questions
* Improved UI and chat-style interface
* Deployment with secure API key management

---

## Author

Venkatesh Gudade
Computer Engineering Undergraduate
Pimpri Chinchwad College of Engineering and Research

LinkedIn: https://www.linkedin.com/in/venkatesh-gudade-85847b259/
GitHub: https://github.com/Venkatesh04-data
