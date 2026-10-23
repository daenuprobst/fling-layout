import matplotlib.pyplot as plt
import networkx as nx
from fling import Fling, FlingStress, FlingVis

# three communities of 80 nodes, densely connected inside and sparsely between
p = [[0.12 if i == j else 0.008 for j in range(3)] for i in range(3)]
graph = nx.stochastic_block_model([80, 80, 80], p, seed=1)
community = [graph.nodes[v]["block"] for v in graph]

# the same network, trained on three different energies
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, cls in zip(axes, (Fling, FlingStress, FlingVis)):
    pos = cls(n_components=2, seed=0).fit_transform(graph)
    nx.draw(graph, pos=dict(enumerate(pos)), ax=ax, node_size=10, node_color=community,
            cmap="tab10", vmin=0, vmax=9, edge_color="0.8", width=0.3)
    ax.set_title(cls.__name__)
plt.show()
