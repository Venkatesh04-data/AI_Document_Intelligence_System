import streamlit as st
import cohere
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Document Intelligence System", layout="wide")

st.title("📄 AI Document Intelligence System")

# API key input
api_key = st.text_input("Enter Cohere API Key", type="password")

# Upload multiple PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

# Question input
question = st.text_input("Ask a question about the documents")


# Load embedding model only once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Extract text from PDFs
def extract_text(files):

    full_text = ""

    for file in files:

        pdf = PdfReader(file)

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                full_text += text + " "

    return full_text


# Sentence-safe chunking (prevents words being cut)
def chunk_text(text, chunk_size=1200):

    sentences = text.split(". ")

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Initialize session state
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
    st.session_state.chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Initialize Cohere
if api_key:
    co = cohere.Client(api_key)


# Process documents only once
if uploaded_files and st.session_state.vector_index is None:

    st.info("Processing documents...")

    text = extract_text(uploaded_files)

    chunks = chunk_text(text)

    model = load_model()

    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    st.session_state.vector_index = index
    st.session_state.chunks = chunks

    st.success("Documents processed successfully. You can now ask questions.")


# Retrieve stored vector DB
index = st.session_state.vector_index
chunks = st.session_state.chunks


# Show previous chat
for chat in st.session_state.chat_history:

    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**AI:** {chat['answer']}")


# Question answering
if question and index is not None:

    model = load_model()

    q_embedding = model.encode([question])

    q_embedding = np.array(q_embedding)

    D, I = index.search(q_embedding, k=3)

    context = ""

    sources = []

    for i in I[0]:

        context += chunks[i] + "\n"

        sources.append(chunks[i])

    prompt = f"""
You are an AI assistant that answers questions using the provided context.

Context:
{context}

Question:
{question}

Answer clearly based only on the context.
"""

    response = co.chat(
        model="command-a-03-2025",
        message=prompt
    )

    answer = response.text

    st.session_state.chat_history.append(
        {"question": question, "answer": answer}
    )

    st.markdown("### 🤖 Answer")

    st.write(answer)

    st.markdown("### 📚 Source Context")

    for i, src in enumerate(sources):

        st.markdown(f"**Source {i+1}**")

        st.info(src)