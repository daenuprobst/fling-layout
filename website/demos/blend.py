import matplotlib.pyplot as plt
import networkx as nx
from fling import FlingBlend

ring = nx.convert_node_labels_to_integers(nx.watts_strogatz_graph(300, 6, 0.02, seed=0))
model = FlingBlend(n_components=2, seed=0).fit(ring)

# one fit, drawn at any blend between stress (0) and neighbour embedding (1)
lams = [0.0, 0.25, 0.5, 0.75, 1.0]
fig, axes = plt.subplots(1, len(lams), figsize=(11, 2.6))
for ax, lam in zip(axes, lams):
    nx.draw(ring, pos=dict(enumerate(model.draw(lam))), ax=ax, node_size=6,
            node_color=range(300), cmap="turbo", edge_color="0.8", width=0.4)
    ax.set_title(f"λ = {lam}")
plt.show()
