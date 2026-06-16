import Pyro5.api

proxy = Pyro5.api.Proxy(
    "PYRO:search.node1@localhost:9090"
)

print(
    proxy.search("machine learning")
)