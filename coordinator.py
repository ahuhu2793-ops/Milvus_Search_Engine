import Pyro5.api
from concurrent.futures import ThreadPoolExecutor

# URI cố định - khớp với objectId đã đăng ký ở node1.py và node2.py
NODE1_URI = "PYRO:search.node1@localhost:9090"
NODE2_URI = "PYRO:search.node2@localhost:9092"

def search_node(uri, query):

    try:
        with Pyro5.api.Proxy(uri) as proxy:
            return proxy.search(query)

    except Exception as e:
        print("Node failed:", e)
        return []

def distributed_search(query):

    with ThreadPoolExecutor(max_workers=2) as executor:

        future1 = executor.submit(
            search_node,
            NODE1_URI,
            query
        )

        future2 = executor.submit(
            search_node,
            NODE2_URI,
            query
        )

        result1 = future1.result()
        result2 = future2.result()

    merged = result1 + result2

    merged.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return merged[:5]

while True:

    query = input("Enter query: ")

    results = distributed_search(query)

    print("\nTop Results")

    for r in results:
        print(r)