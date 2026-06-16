from sentence_transformers import SentenceTransformer

# Danh sách tài liệu mẫu
documents = [
    "Introduction to AI",
    "Machine Learning Basics",
    "Deep Learning Techniques",
    "Natural Language Processing",
    "Computer Vision"
]

print("Loading model...")

# Model sinh embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(documents)

print("Number of documents:", len(documents))
print("Embedding dimension:", len(embeddings[0]))

for i, doc in enumerate(documents):
    print(f"\nDocument {i+1}:")
    print(doc)
    print("Vector length:", len(embeddings[i]))