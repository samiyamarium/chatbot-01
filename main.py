import os
import streamlit as st
from qdrant_client import QdrantClient
from google import genai

# Load secrets from Streamlit
# ---------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_COLLECTION = st.secrets["QDRANT_COLLECTION"]

# ---------------------------
# Initialize clients
# ---------------------------
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------
# Retriever (NO embeddings)
# ---------------------------
def retriever_agent(question: str):
    hits, _ = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=10
    )

    matched = []
    for point in hits:
        text = point.payload.get("text", "")
        if any(word.lower() in text.lower() for word in question.split()):
            matched.append(point.payload)

    return matched[:3]

# ---------------------------
# Gemini LLM
# ---------------------------
def llm_agent(question: str, context_blocks):
    if not context_blocks:
        return "The book does not contain this information."

    context = "\n\n".join(c["text"] for c in context_blocks)

    prompt = f"""
You are an academic book assistant.

Rules:
- Answer strictly from the provided book content
- If the answer is not present, say:
  "The book does not contain this information."

Book Content:
{context}

Question:
{question}
"""

    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Book Chatbot", layout="centered")

st.title("📘 Book Question Answering Bot")
st.write("Ask questions strictly based on the book content.")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the book..."):
            context_blocks = retriever_agent(question)
            answer = llm_agent(question, context_blocks)

            sources = list({
                c.get("chapter_title", "Unknown")
                for c in context_blocks
            })

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for src in sources:
                st.markdown(f"- {src}")
