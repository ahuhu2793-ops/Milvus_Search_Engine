from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530"
)

for name in ["documents_1", "documents_2"]:

    if client.has_collection(name):
        client.drop_collection(name)

    client.create_collection(
        collection_name=name,
        dimension=384
    )

    print(f"{name} created")