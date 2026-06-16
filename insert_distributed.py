from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient(
    uri="http://localhost:19530"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

documents = []

topics = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Computer Vision",
    "Natural Language Processing"
]

for i in range(100):
    documents.append(
        f"{topics[i % 5]} document number {i}"
    )

embeddings = model.encode(documents)

data1 = []
data2 = []

for i, emb in enumerate(embeddings):

    item = {
        "id": i,
        "vector": emb.tolist()
    }

    if i < 50:
        data1.append(item)
    else:
        data2.append(item)

client.insert(
    collection_name="documents_1",
    data=data1
)

client.insert(
    collection_name="documents_2",
    data=data2
)

print("Inserted into documents_1:", len(data1))
print("Inserted into documents_2:", len(data2))