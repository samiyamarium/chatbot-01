import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from google import genai

load_dotenv()

# -------------------
# Qdrant
# -------------------
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION = os.getenv("QDRANT_COLLECTION")

# -------------------
# Gemini
# -------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------
# Retriever Agent
# -------------------
def retriever_agent(question: str):
    """
    Retrieves relevant chapters using keyword match (no embeddings)
    """
    hits = qdrant.scroll(
        collection_name=COLLECTION,
        limit=50
    )[0]

    matched = []
    for point in hits:
        text = point.payload.get("text", "")
        if question.lower() in text.lower():
            matched.append(point.payload)

    return matched[:3]


# -------------------
# LLM Agent
# -------------------
def llm_agent(question: str, context_blocks):
    if not context_blocks:
        return "The book does not contain this information."

    context = "\n\n".join([c["text"] for c in context_blocks])

    prompt = f"""
You are an academic book assistant .

Rules:
- Answer strictly from the provided book content
- If answer is not present, say:
  "The book does not contain this information."

Book Content:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# -------------------
# TRIAGE AGENT
# -------------------
def answer_question(question: str):
    context_blocks = retriever_agent(question)
    answer = llm_agent(question, context_blocks)

    sources = list({
        c.get("chapter_title", "Unknown")
        for c in context_blocks
    })

    return answer, sources
