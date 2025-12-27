import streamlit as st
from qdrant_client import QdrantClient
from google import genai

# -------------------------------------------------
# Streamlit security hardening
# -------------------------------------------------
st.set_option("client.showErrorDetails", False)

# -------------------------------------------------
# Load secrets (NEVER print these)
# -------------------------------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_COLLECTION = st.secrets["QDRANT_COLLECTION"]

# -------------------------------------------------
# Initialize Qdrant client (server-side only)
# -------------------------------------------------
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# -------------------------------------------------
# Initialize Gemini (SAFE SDK)
# -------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.2,
        "max_output_tokens": 512,
    },
)

# -------------------------------------------------
# Retriever (keyword-based, no embeddings)
# -------------------------------------------------
def retriever_agent(question: str):
    try:
        points, _ = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10,
        )
    except Exception:
        return []

    matched = []
    question_words = question.lower().split()

    for point in points:
        text = point.payload.get("text", "")
        if any(word in text.lower() for word in question_words):
            matched.append(point.payload)

    return matched[:3]

# -------------------------------------------------
# LLM Agent (STRICT book-only answers)
# -------------------------------------------------
def llm_agent(question: str, context_blocks):
    if not context_blocks:
        return "The book does not contain this information."

    context = "\n\n".join(block["text"] for block in context_blocks)

    prompt = f"""
You are an academic book assistant.

Rules:
- Answer strictly from the provided book content
- Do NOT add external knowledge
- If the answer is not present, say exactly:
  "The book does not contain this information."

Book Content:
{context}

Question:
{question}
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return "The book does not contain this information."

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="Book Chatbot by Samiya Marium",
    layout="centered",
)

st.title("📘 Physical-AI & Humanoid Robotics")
st.caption("Answers are strictly based on the book content.")

question = st.text_input("Enter your question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the book..."):
            context_blocks = retriever_agent(question)
            answer = llm_agent(question, context_blocks)

            sources = sorted(
                {
                    block.get("chapter_title", "Unknown")
                    for block in context_blocks
                }
            )

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for src in sources:
                st.markdown(f"- {src}")
