from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np
import Pyro5.api
import time

app = Flask(__name__)

# ============================================================
# Load embedding model
# ============================================================
print("[INFO] Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("[INFO] Model loaded!")

# ============================================================
# Pyro5 Node URIs  (khớp node1.py / node2.py)
# ============================================================
NODES = {
    "Node 1": {
        "uri":        "PYRO:search.node1@localhost:9090",
        "collection": "documents_1",
        "port":       9090,
        "docs":       "ID 0-49",
    },
    "Node 2": {
        "uri":        "PYRO:search.node2@localhost:9092",
        "collection": "documents_2",
        "port":       9092,
        "docs":       "ID 50-99",
    },
}

# ============================================================
# Document metadata store  (30 học thuật tài liệu)
# ============================================================
documents_db = [
    # ── Artificial Intelligence ──────────────────────────────
    {"id": 0,  "title": "Artificial Intelligence: A Modern Approach",    "category": "Artificial Intelligence", "author": "Stuart Russell & Peter Norvig", "year": 2021, "pages": 1132, "description": "Giáo trình AI nổi tiếng nhất thế giới, bao phủ toàn diện các lĩnh vực của trí tuệ nhân tạo từ cơ bản đến nâng cao."},
    {"id": 1,  "title": "AI Ethics and Responsible Technology",          "category": "Artificial Intelligence", "author": "Virginia Dignum",              "year": 2022, "pages": 280,  "description": "Phân tích đạo đức trong phát triển và ứng dụng AI, các nguyên tắc thiết kế AI có trách nhiệm."},
    {"id": 2,  "title": "Planning Algorithms for AI Systems",            "category": "Artificial Intelligence", "author": "Steven LaValle",               "year": 2020, "pages": 842,  "description": "Các thuật toán lập kế hoạch trong AI, từ tìm kiếm đồ thị đến lập kế hoạch trong không gian liên tục."},
    {"id": 3,  "title": "Knowledge Representation in AI",               "category": "Artificial Intelligence", "author": "Ronald Brachman",              "year": 2022, "pages": 430,  "description": "Phương pháp biểu diễn tri thức trong hệ thống AI: ontology, logic, mạng ngữ nghĩa."},
    {"id": 4,  "title": "Game Theory and Multi-Agent Systems",          "category": "Artificial Intelligence", "author": "Yoav Shoham",                  "year": 2021, "pages": 512,  "description": "Lý thuyết trò chơi ứng dụng trong thiết kế hệ thống đa tác nhân và các thuật toán Nash equilibrium."},
    {"id": 5,  "title": "Expert Systems and Knowledge Engineering",     "category": "Artificial Intelligence", "author": "Edward Feigenbaum",            "year": 2020, "pages": 360,  "description": "Xây dựng hệ thống chuyên gia, suy diễn tri thức và ứng dụng trong y tế, pháp lý, tài chính."},
    # ── Machine Learning ─────────────────────────────────────
    {"id": 6,  "title": "Introduction to Machine Learning",             "category": "Machine Learning",        "author": "Ethem Alpaydin",               "year": 2020, "pages": 640,  "description": "Khởi đầu hoàn hảo về học máy: hồi quy, phân loại, clustering, và đánh giá mô hình."},
    {"id": 7,  "title": "Pattern Recognition and Machine Learning",     "category": "Machine Learning",        "author": "Christopher Bishop",           "year": 2021, "pages": 738,  "description": "Tiếp cận xác suất trong nhận dạng mẫu và học máy, bao gồm Bayesian methods và Gaussian processes."},
    {"id": 8,  "title": "The Elements of Statistical Learning",        "category": "Machine Learning",        "author": "Hastie, Tibshirani, Friedman", "year": 2022, "pages": 764,  "description": "Nền tảng thống kê học máy: cây quyết định, SVM, boosting, neural networks và regularization."},
    {"id": 9,  "title": "Reinforcement Learning: An Introduction",     "category": "Machine Learning",        "author": "Sutton & Barto",               "year": 2021, "pages": 552,  "description": "Sách giáo khoa chuẩn về học tăng cường: MDP, Q-learning, policy gradient và ứng dụng trong game AI."},
    {"id": 10, "title": "Hands-On Machine Learning with Scikit-Learn", "category": "Machine Learning",        "author": "Aurélien Géron",               "year": 2023, "pages": 851,  "description": "Thực hành học máy với Python, Scikit-Learn và TensorFlow qua các dự án thực tế từng bước."},
    {"id": 11, "title": "Feature Engineering for Machine Learning",    "category": "Machine Learning",        "author": "Alice Zheng",                  "year": 2022, "pages": 248,  "description": "Kỹ thuật trích xuất và biến đổi đặc trưng để tối ưu hiệu suất mô hình học máy."},
    # ── Deep Learning ────────────────────────────────────────
    {"id": 12, "title": "Deep Learning",                                "category": "Deep Learning",           "author": "Goodfellow, Bengio, Courville", "year": 2021, "pages": 800,  "description": "Kinh điển về deep learning: MLP, CNN, RNN, regularization, optimization và generative models."},
    {"id": 13, "title": "Neural Networks and Deep Learning",           "category": "Deep Learning",           "author": "Michael Nielsen",              "year": 2022, "pages": 460,  "description": "Giải thích trực quan về mạng nơ-ron và học sâu, backpropagation từ đầu với Python thuần."},
    {"id": 14, "title": "Attention Is All You Need - Transformer",    "category": "Deep Learning",           "author": "Vaswani et al. (Google)",      "year": 2022, "pages": 15,   "description": "Bài báo gốc về kiến trúc Transformer, nền tảng của GPT, BERT và mọi mô hình ngôn ngữ lớn hiện đại."},
    {"id": 15, "title": "Generative Adversarial Networks (GANs)",     "category": "Deep Learning",           "author": "Ian Goodfellow",               "year": 2021, "pages": 180,  "description": "Lý thuyết và thực hành GAN: từ vanilla GAN đến StyleGAN, CycleGAN và ứng dụng tạo ảnh."},
    {"id": 16, "title": "Graph Neural Networks: Foundations",         "category": "Deep Learning",           "author": "Yao Ma & Jiliang Tang",        "year": 2023, "pages": 424,  "description": "GNN cho dữ liệu đồ thị: node classification, link prediction, graph classification và ứng dụng mạng xã hội."},
    {"id": 17, "title": "Convolutional Neural Networks Architecture", "category": "Deep Learning",           "author": "Kaiming He",                   "year": 2022, "pages": 210,  "description": "Các kiến trúc CNN từ LeNet, AlexNet, VGG đến ResNet, DenseNet và EfficientNet."},
    # ── NLP ──────────────────────────────────────────────────
    {"id": 18, "title": "Natural Language Processing with Python",    "category": "NLP",                     "author": "Steven Bird, Ewan Klein",      "year": 2022, "pages": 504,  "description": "Xử lý ngôn ngữ tự nhiên với Python và NLTK: tokenization, POS tagging, parsing và sentiment analysis."},
    {"id": 19, "title": "Speech and Language Processing",             "category": "NLP",                     "author": "Jurafsky & Martin",            "year": 2023, "pages": 654,  "description": "Toàn diện về NLP và xử lý tiếng nói: từ n-gram đến Transformer, machine translation, QA systems."},
    {"id": 20, "title": "BERT: Pre-training Deep Bidirectional Transformers", "category": "NLP",            "author": "Devlin et al. (Google)",       "year": 2021, "pages": 16,   "description": "Bài báo BERT – mô hình ngôn ngữ pre-train hai chiều, đặt nền móng cho ChatGPT và các LLM hiện đại."},
    {"id": 21, "title": "Text Mining and Information Extraction",     "category": "NLP",                     "author": "Bing Liu",                     "year": 2022, "pages": 560,  "description": "Khai thác thông tin từ văn bản: NER, relation extraction, event extraction và opinion mining."},
    {"id": 22, "title": "Sentiment Analysis and Opinion Mining",      "category": "NLP",                     "author": "Bing Liu",                     "year": 2021, "pages": 312,  "description": "Phân tích cảm xúc và khai thác ý kiến từ đánh giá sản phẩm, mạng xã hội và báo chí."},
    {"id": 23, "title": "Machine Translation with Neural Networks",   "category": "NLP",                     "author": "Philipp Koehn",                "year": 2022, "pages": 385,  "description": "Dịch máy nơ-ron: seq2seq, attention mechanism, và các hệ thống dịch thuật hiện đại."},
    # ── Computer Vision ──────────────────────────────────────
    {"id": 24, "title": "Computer Vision: Algorithms and Applications","category": "Computer Vision",         "author": "Richard Szeliski",             "year": 2022, "pages": 979,  "description": "Toàn diện về thị giác máy tính: image processing, feature detection, stereo vision và 3D reconstruction."},
    {"id": 25, "title": "Digital Image Processing",                   "category": "Computer Vision",         "author": "Gonzalez & Woods",             "year": 2021, "pages": 1026, "description": "Xử lý ảnh số nền tảng: histogram, filtering, morphology, segmentation và image compression."},
    {"id": 26, "title": "Object Detection and Recognition with YOLO", "category": "Computer Vision",         "author": "Joseph Redmon",                "year": 2022, "pages": 42,   "description": "Nhận diện đối tượng thời gian thực với YOLO: từ YOLOv1 đến YOLOv8, anchor boxes và NMS."},
    {"id": 27, "title": "Medical Image Analysis with Deep Learning",  "category": "Computer Vision",         "author": "Dinggang Shen",                "year": 2023, "pages": 540,  "description": "Ứng dụng deep learning trong phân tích ảnh y tế: MRI, CT scan, X-ray, phân đoạn khối u."},
    {"id": 28, "title": "3D Computer Vision and Point Cloud Learning","category": "Computer Vision",         "author": "Charles Qi",                   "year": 2022, "pages": 280,  "description": "Thị giác 3D: depth estimation, point cloud processing, PointNet và ứng dụng xe tự lái."},
    {"id": 29, "title": "Video Understanding and Action Recognition", "category": "Computer Vision",         "author": "Li Fei-Fei",                   "year": 2023, "pages": 320,  "description": "Phân tích video: optical flow, temporal modeling, action recognition và video captioning."},
]

# Pre-compute embeddings
print("[INFO] Pre-computing document embeddings...")
texts_for_embed = [f"{d['title']} {d['category']} {d['description']}" for d in documents_db]
doc_embeddings  = model.encode(texts_for_embed, show_progress_bar=False)
for i, doc in enumerate(documents_db):
    doc["_vector"] = doc_embeddings[i]

DOC_BY_ID = {d["id"]: d for d in documents_db}
CATEGORY_ICONS = {
    "Artificial Intelligence": "🤖",
    "Machine Learning":        "📊",
    "Deep Learning":           "🧠",
    "NLP":                     "💬",
    "Computer Vision":         "👁️",
}
print(f"[INFO] Ready! {len(documents_db)} documents indexed.")


# ============================================================
# Helpers
# ============================================================
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _doc_payload(doc, score):
    return {
        "id":          doc["id"],
        "title":       doc["title"],
        "category":    doc["category"],
        "icon":        CATEGORY_ICONS.get(doc["category"], "📄"),
        "author":      doc["author"],
        "year":        doc["year"],
        "pages":       doc["pages"],
        "description": doc["description"],
        "score":       round(score * 100, 1),
    }


# ── Pyro5 single-node search ──────────────────────────────
def search_one_node(node_name, node_info, query_vector_list, query_raw):
    """
    Gửi truy vấn đến một Pyro5 node.
    Trả về dict: { ok, results, latency_ms, error }

    Fault-Tolerance:
      - Nếu node OFFLINE → except bắt lỗi → trả ok=False, results=[]
      - Hệ thống tiếp tục merge kết quả từ node còn lại
    """
    t0 = time.time()
    try:
        with Pyro5.api.Proxy(node_info["uri"]) as proxy:
            proxy._pyroTimeout = 3          # timeout 3 giây
            raw = proxy.search(query_raw)   # node trả list[{id, score, node}]
        latency = round((time.time() - t0) * 1000)

        # Enrich với metadata
        results = []
        for hit in raw:
            doc = DOC_BY_ID.get(hit["id"])
            if doc:
                results.append(_doc_payload(doc, hit["score"]))

        return {"ok": True, "results": results, "latency_ms": latency, "error": None}

    except Exception as e:
        latency = round((time.time() - t0) * 1000)
        print(f"[FAULT] {node_name} failed: {e}")
        # ← Đây là Fault Tolerance: node lỗi → trả list rỗng, không crash
        return {"ok": False, "results": [], "latency_ms": latency, "error": str(e)}


# ── Local cosine fallback (khi cả 2 node offline) ────────
def search_local_fallback(query, top_k=8):
    qvec = model.encode(query)
    scored = [(cosine_similarity(qvec, d["_vector"]), d) for d in documents_db]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [_doc_payload(d, s) for s, d in scored[:top_k]]


# ── Distributed search  (coordinator logic) ───────────────
def distributed_search(query, top_k=8):
    """
    Gửi truy vấn song song đến Node1 & Node2 (ThreadPoolExecutor).
    Merge → sort theo score → top_k.
    Nếu một node chết → kết quả từ node còn lại vẫn trả về bình thường.
    """
    query_vec = model.encode(query).tolist()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            name: executor.submit(search_one_node, name, info, query_vec, query)
            for name, info in NODES.items()
        }
        node_results = {name: f.result() for name, f in futures.items()}

    # Merge từ các node thành công
    merged = []
    for name, res in node_results.items():
        for doc in res["results"]:
            doc["source_node"] = name          # gắn nhãn node nguồn
            merged.append(doc)

    merged.sort(key=lambda x: x["score"], reverse=True)

    # Nếu cả 2 node đều offline → fallback local
    all_failed = all(not r["ok"] for r in node_results.values())
    if all_failed:
        fallback = search_local_fallback(query, top_k)
        for d in fallback:
            d["source_node"] = "Local Fallback"
        return fallback, node_results, True

    return merged[:top_k], node_results, False


# ── Node health-check ─────────────────────────────────────
def check_node_health(node_name, node_info):
    t0 = time.time()
    try:
        with Pyro5.api.Proxy(node_info["uri"]) as proxy:
            proxy._pyroTimeout = 2
            proxy._pyroBind()
        return {
            "name":       node_name,
            "status":     "online",
            "latency_ms": round((time.time() - t0) * 1000),
            "port":       node_info["port"],
            "collection": node_info["collection"],
            "docs":       node_info["docs"],
        }
    except Exception as e:
        return {
            "name":       node_name,
            "status":     "offline",
            "latency_ms": None,
            "port":       node_info["port"],
            "collection": node_info["collection"],
            "docs":       node_info["docs"],
            "error":      str(e)[:80],
        }


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    categories = ["All"] + list(CATEGORY_ICONS.keys())
    stats = {
        "total_docs":    len(documents_db),
        "total_cats":    len(CATEGORY_ICONS),
        "total_authors": len(set(d["author"] for d in documents_db)),
    }
    return render_template("index.html", categories=categories, stats=stats)


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": [], "total": 0, "query": "", "node_details": {}})

    results, node_results, used_fallback = distributed_search(query, top_k=8)

    # Tóm tắt trạng thái từng node để hiển thị UI
    node_summary = {}
    for name, res in node_results.items():
        node_summary[name] = {
            "ok":         res["ok"],
            "count":      len(res["results"]),
            "latency_ms": res["latency_ms"],
            "error":      res.get("error"),
        }

    return jsonify({
        "results":      results,
        "total":        len(results),
        "query":        query,
        "node_details": node_summary,
        "fallback":     used_fallback,
    })


@app.route("/api/node-status")
def api_node_status():
    """Health-check cả 2 node — gọi mỗi 5 giây từ frontend."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            name: executor.submit(check_node_health, name, info)
            for name, info in NODES.items()
        }
        statuses = [f.result() for f in futures.values()]
    return jsonify({"nodes": statuses})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
