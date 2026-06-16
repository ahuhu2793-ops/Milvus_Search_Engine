from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient(
    uri="http://localhost:19530"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

query = "Artificial Intelligence"

query_vector = model.encode(query)

results = client.search(
    collection_name="documents",
    data=[query_vector.tolist()],
    limit=5
)

print("\nTop 5 Results:\n")

for hit in results[0]:
    print(hit)