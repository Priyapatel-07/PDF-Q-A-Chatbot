#streamlit run app.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("Chat with your PDF")

# Load local embedding model once
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    # Extract text
    reader = PdfReader(pdf)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(raw_text)

    # Embed all chunks locally
    @st.cache_data
    def embed_chunks(text_chunks):
        return embedder.encode(text_chunks)

    chunk_vecs = embed_chunks(tuple(chunks))

    st.success(f" PDF loaded! ({len(chunks)} chunks indexed)")

    question = st.text_input("Ask a question about the PDF:")

    if question:
        # Embed question locally
        q_vec = embedder.encode(question)

        # Cosine similarity to find top chunks
        scores = np.dot(chunk_vecs, q_vec) / (
            np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(q_vec)
        )
        top_indices = np.argsort(scores)[-3:][::-1]
        context = "\n\n".join([chunks[i] for i in top_indices])

        # Gemini answers using the context
        gemini = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""Use the context below to answer the question.
If the answer is not in the context, say "I don't know based on this PDF."

Context:
{context}

Question: {question}

Answer:"""
        response = gemini.generate_content(prompt)

        st.write("### Answer:")
        st.write(response.text)