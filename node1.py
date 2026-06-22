"""
Node 1 – Pyro5 RPC Search Server  (port 9090)
Quản lý tài liệu ID 0-14 (nửa đầu kho tài liệu)
Không yêu cầu Milvus — dùng local SentenceTransformer vector search.
"""
import Pyro5.api
import Pyro5.server
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Load model ─────────────────────────────────────────────
print("[Node1] Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("[Node1] Model loaded!")

# ── Tài liệu thuộc Node 1 (ID 0 - 14) ────────────────────
documents = [
    {"id": 0,  "title": "Artificial Intelligence: A Modern Approach",    "category": "Artificial Intelligence", "description": "Giáo trình AI nổi tiếng nhất thế giới, bao phủ toàn diện các lĩnh vực của trí tuệ nhân tạo."},
    {"id": 1,  "title": "AI Ethics and Responsible Technology",          "category": "Artificial Intelligence", "description": "Phân tích đạo đức trong phát triển và ứng dụng AI."},
    {"id": 2,  "title": "Planning Algorithms for AI Systems",            "category": "Artificial Intelligence", "description": "Các thuật toán lập kế hoạch trong AI."},
    {"id": 3,  "title": "Knowledge Representation in AI",               "category": "Artificial Intelligence", "description": "Phương pháp biểu diễn tri thức trong hệ thống AI."},
    {"id": 4,  "title": "Game Theory and Multi-Agent Systems",          "category": "Artificial Intelligence", "description": "Lý thuyết trò chơi ứng dụng trong thiết kế hệ thống đa tác nhân."},
    {"id": 5,  "title": "Expert Systems and Knowledge Engineering",     "category": "Artificial Intelligence", "description": "Xây dựng hệ thống chuyên gia, suy diễn tri thức."},
    {"id": 6,  "title": "Introduction to Machine Learning",             "category": "Machine Learning",        "description": "Khởi đầu hoàn hảo về học máy: hồi quy, phân loại, clustering."},
    {"id": 7,  "title": "Pattern Recognition and Machine Learning",     "category": "Machine Learning",        "description": "Tiếp cận xác suất trong nhận dạng mẫu và học máy."},
    {"id": 8,  "title": "The Elements of Statistical Learning",        "category": "Machine Learning",        "description": "Nền tảng thống kê học máy: cây quyết định, SVM, boosting."},
    {"id": 9,  "title": "Reinforcement Learning: An Introduction",     "category": "Machine Learning",        "description": "Sách giáo khoa chuẩn về học tăng cường: MDP, Q-learning."},
    {"id": 10, "title": "Hands-On Machine Learning with Scikit-Learn", "category": "Machine Learning",        "description": "Thực hành học máy với Python, Scikit-Learn và TensorFlow."},
    {"id": 11, "title": "Feature Engineering for Machine Learning",    "category": "Machine Learning",        "description": "Kỹ thuật trích xuất và biến đổi đặc trưng."},
    {"id": 12, "title": "Deep Learning",                                "category": "Deep Learning",           "description": "Kinh điển về deep learning: MLP, CNN, RNN."},
    {"id": 13, "title": "Neural Networks and Deep Learning",           "category": "Deep Learning",           "description": "Giải thích trực quan về mạng nơ-ron và học sâu."},
    {"id": 14, "title": "Attention Is All You Need - Transformer",    "category": "Deep Learning",           "description": "Bài báo gốc về kiến trúc Transformer."},
]

# Pre-compute embeddings
print("[Node1] Pre-computing embeddings...")
texts = [f"{d['title']} {d['category']} {d['description']}" for d in documents]
embeddings = model.encode(texts, show_progress_bar=False)
for i, doc in enumerate(documents):
    doc["_vector"] = embeddings[i]
print(f"[Node1] Indexed {len(documents)} documents (ID 0-14)")


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ── Pyro5 RPC Service ──────────────────────────────────────
@Pyro5.api.expose
class SearchNode:

    def search(self, query):
        """Tìm kiếm semantic trong tập tài liệu của Node 1."""
        query_vector = model.encode(query)

        scored = []
        for doc in documents:
            score = cosine_sim(query_vector, doc["_vector"])
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        output = []
        for score, doc in scored[:5]:
            output.append({
                "node":  "Node 1",
                "id":    doc["id"],
                "score": round(score, 4),
            })

        print(f"[Node1] Query: '{query}' → {len(output)} results")
        return output


# ── Start Daemon ────────────────────────────────────────────
daemon = Pyro5.server.Daemon(port=9090)
uri = daemon.register(SearchNode, objectId="search.node1")

print(f"[Node1] ✅ Đang chạy trên port 9090")
print(f"[Node1] URI: {uri}")
print(f"[Node1] Nhấn Ctrl+C để tắt (mô phỏng node offline)")

daemon.requestLoop()