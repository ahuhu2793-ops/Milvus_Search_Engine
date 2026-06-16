documents = []

for i in range(100):
    documents.append(
        f"Artificial Intelligence document number {i}"
    )

print("Tổng số tài liệu:", len(documents))

print("\n5 tài liệu đầu tiên:")
for doc in documents[:5]:
    print(doc)