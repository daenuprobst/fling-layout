import matplotlib.pyplot as plt
import networkx as nx
from fling import FlingStress

# a tree in which every node has three children, five levels deep (364 nodes)
graph = nx.balanced_tree(3, 5)
pos = dict(enumerate(FlingStress(n_components=2, seed=0).fit_transform(graph)))

# nodes coloured by their depth, the number of steps from the root
depth = nx.shortest_path_length(graph, 0)
nx.draw_networkx_edges(graph, pos, edge_color="0.75", width=0.6)
nodes = nx.draw_networkx_nodes(graph, pos, node_size=14, cmap="viridis",
                               node_color=[depth[v] for v in graph])
plt.colorbar(nodes, label="depth from the root", ticks=range(6), shrink=0.6)
plt.gca().set_aspect("equal")
plt.axis("off")
plt.show()
