import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fling import FlingVis

# three communities of 80 nodes; hold 36 of them back from training
p = [[0.12 if i == j else 0.008 for j in range(3)] for i in range(3)]
graph = nx.stochastic_block_model([80, 80, 80], p, seed=1)
new = set(np.random.default_rng(0).choice(240, 36, replace=False).tolist())
seen = graph.subgraph([v for v in graph if v not in new])

# train on the 204 nodes it sees, then place all 240 with the trained model, no refitting
model = FlingVis(n_components=2, seed=0).fit(seen)
pos = dict(zip(graph.nodes(), model.transform(graph)))

# new nodes outlined, coloured by the community they belong to
colours = ["tab:blue", "tab:orange", "tab:green"]
old = [v for v in graph if v not in new]
nx.draw_networkx_edges(graph, pos, edge_color="0.85", width=0.3)
nx.draw_networkx_nodes(graph, pos, nodelist=old, node_size=8, alpha=0.35,
                       node_color=[colours[graph.nodes[v]["block"]] for v in old])
nx.draw_networkx_nodes(graph, pos, nodelist=sorted(new), node_size=30, edgecolors="black",
                       linewidths=0.6, node_color=[colours[graph.nodes[v]["block"]] for v in sorted(new)])
plt.axis("off")
plt.show()
