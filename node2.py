import Pyro5.api
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient(
    uri="http://localhost:19530"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

@Pyro5.api.expose
class SearchNode:

    def search(self, query):

        query_vector = model.encode(query)

        results = client.search(
            collection_name="documents_2",
            data=[query_vector.tolist()],
            limit=3
        )

        output = []

        for hit in results[0]:
            output.append({
                "node": "Node2",
                "id": hit["id"],
                "score": hit["distance"]
            })

        return output

daemon = Pyro5.server.Daemon(port=9092)

# Đăng ký class với tên cố định để URI không đổi mỗi lần restart
uri = daemon.register(SearchNode, objectId="search.node2")

print("Node2 đang chạy trên port 9092")
print(f"URI của bạn: {uri}")

daemon.requestLoop()