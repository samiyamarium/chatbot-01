import os
import glob
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBEDDING_MODEL)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create collection if it doesn't exist
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vector_size=384,
        distance="Cosine"
    )

# Ingest all chapters
chapters = glob.glob("books/*.md")
for i, file_path in enumerate(chapters):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    vector = embedder.encode(text).tolist()
    client.upsert(
        collection_name=COLLECTION,
        points=[{
            "id": i,
            "vector": vector,
            "payload": {"text": text, "chapter_title": os.path.basename(file_path)}
        }]
    )

print(f"✅ Ingested {len(chapters)} chapters into Qdrant")
