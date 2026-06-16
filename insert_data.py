from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient(
    uri="http://localhost:19530"
)

documents = []

for i in range(100):
    documents.append(
        f"Artificial Intelligence document number {i}"
    )

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(documents)

data = []

for i, vector in enumerate(embeddings):
    data.append({
        "id": i,
        "vector": vector.tolist()
    })

client.insert(
    collection_name="documents",
    data=data
)

print("Inserted", len(data), "documents")