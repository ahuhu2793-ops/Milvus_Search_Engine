"""
Node 2 – Pyro5 RPC Search Server  (port 9092)
Quản lý tài liệu ID 15-29 (nửa sau kho tài liệu)
Không yêu cầu Milvus — dùng local SentenceTransformer vector search.
"""
import Pyro5.api
import Pyro5.server
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Load model ─────────────────────────────────────────────
print("[Node2] Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("[Node2] Model loaded!")

# ── Tài liệu thuộc Node 2 (ID 15 - 29) ───────────────────
documents = [
    {"id": 15, "title": "Generative Adversarial Networks (GANs)",     "category": "Deep Learning",           "description": "Lý thuyết và thực hành GAN: từ vanilla GAN đến StyleGAN, CycleGAN."},
    {"id": 16, "title": "Graph Neural Networks: Foundations",         "category": "Deep Learning",           "description": "GNN cho dữ liệu đồ thị: node classification, link prediction."},
    {"id": 17, "title": "Convolutional Neural Networks Architecture", "category": "Deep Learning",           "description": "Các kiến trúc CNN từ LeNet, AlexNet, VGG đến ResNet, EfficientNet."},
    {"id": 18, "title": "Natural Language Processing with Python",    "category": "NLP",                     "description": "Xử lý ngôn ngữ tự nhiên với Python và NLTK: tokenization, POS tagging."},
    {"id": 19, "title": "Speech and Language Processing",             "category": "NLP",                     "description": "Toàn diện về NLP và xử lý tiếng nói: từ n-gram đến Transformer."},
    {"id": 20, "title": "BERT: Pre-training Deep Bidirectional Transformers", "category": "NLP",            "description": "Bài báo BERT – mô hình ngôn ngữ pre-train hai chiều."},
    {"id": 21, "title": "Text Mining and Information Extraction",     "category": "NLP",                     "description": "Khai thác thông tin từ văn bản: NER, relation extraction."},
    {"id": 22, "title": "Sentiment Analysis and Opinion Mining",      "category": "NLP",                     "description": "Phân tích cảm xúc và khai thác ý kiến từ đánh giá sản phẩm."},
    {"id": 23, "title": "Machine Translation with Neural Networks",   "category": "NLP",                     "description": "Dịch máy nơ-ron: seq2seq, attention mechanism."},
    {"id": 24, "title": "Computer Vision: Algorithms and Applications","category": "Computer Vision",         "description": "Toàn diện về thị giác máy tính: image processing, feature detection."},
    {"id": 25, "title": "Digital Image Processing",                   "category": "Computer Vision",         "description": "Xử lý ảnh số nền tảng: histogram, filtering, morphology, segmentation."},
    {"id": 26, "title": "Object Detection and Recognition with YOLO", "category": "Computer Vision",         "description": "Nhận diện đối tượng thời gian thực với YOLO."},
    {"id": 27, "title": "Medical Image Analysis with Deep Learning",  "category": "Computer Vision",         "description": "Ứng dụng deep learning trong phân tích ảnh y tế: MRI, CT scan."},
    {"id": 28, "title": "3D Computer Vision and Point Cloud Learning","category": "Computer Vision",         "description": "Thị giác 3D: depth estimation, point cloud processing, PointNet."},
    {"id": 29, "title": "Video Understanding and Action Recognition", "category": "Computer Vision",         "description": "Phân tích video: optical flow, temporal modeling, action recognition."},
]

# Pre-compute embeddings
print("[Node2] Pre-computing embeddings...")
texts = [f"{d['title']} {d['category']} {d['description']}" for d in documents]
embeddings = model.encode(texts, show_progress_bar=False)
for i, doc in enumerate(documents):
    doc["_vector"] = embeddings[i]
print(f"[Node2] Indexed {len(documents)} documents (ID 15-29)")


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ── Pyro5 RPC Service ──────────────────────────────────────
@Pyro5.api.expose
class SearchNode:

    def search(self, query):
        """Tìm kiếm semantic trong tập tài liệu của Node 2."""
        query_vector = model.encode(query)

        scored = []
        for doc in documents:
            score = cosine_sim(query_vector, doc["_vector"])
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        output = []
        for score, doc in scored[:5]:
            output.append({
                "node":  "Node 2",
                "id":    doc["id"],
                "score": round(score, 4),
            })

        print(f"[Node2] Query: '{query}' → {len(output)} results")
        return output


# ── Start Daemon ────────────────────────────────────────────
daemon = Pyro5.server.Daemon(port=9092)
uri = daemon.register(SearchNode, objectId="search.node2")

print(f"[Node2] ✅ Đang chạy trên port 9092")
print(f"[Node2] URI: {uri}")
print(f"[Node2] Nhấn Ctrl+C để tắt (mô phỏng node offline)")

daemon.requestLoop()