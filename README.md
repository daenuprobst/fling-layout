# fling-layout

Graph layout by fitting a function of node features instead of a table of coordinates. This is the
code behind the paper. The library is in `fling` and the scripts that build every table and figure
are in `scripts`.

## Install

Requires python 3.10 or newer. The package is `fling-layout` and the module it installs is `fling`.

```
uv pip install -e ".[all]"
```

To install it somewhere else, build a wheel first.

```
uv build
uv pip install dist/fling_layout-0.1.0-py3-none-any.whl
```

numpy, scipy, networkx, torch, scikit-learn, s_gd2 and fa2-modified come with it. Two extras are
optional. `figures` adds matplotlib, which only `fling.figstyle` needs, and `large` adds datashader,
pandas, colorcet and pynndescent for the million node scripts. `all` is both.

The benchmark graphs are not in the wheel. An installed copy finds them through the `FLING_DATA`
environment variable, and a clone falls back to the `data` directory next to the package.

## Library

Each node carries its diffusion potential to a set of landmarks. A small network maps that feature
vector to a position, and the weights are trained on a layout energy. The three variants differ only
in which energy.

```python
import networkx as nx
from fling import FlingStress

graph = nx.convert_node_labels_to_integers(nx.les_miserables_graph())
pos = FlingStress(n_components=2, seed=0).fit_transform(graph)
```

An added node is placed by one forward pass, with no refitting.

```python
pos_all = FlingStress(n_components=2, seed=0).fit(graph).transform(larger_graph)
```

`FlingBlend` conditions the field on a weight between two energies, so one fit answers any weight.

```python
from fling import FlingBlend

model = FlingBlend(n_components=2, seed=0).fit(graph)
early = model.draw(0.0)
late = model.draw(1.0)
```

### Variants

| class | energy | default iters |
|---|---|---|
| `Fling` | all pairs stress by majorisation, with a second field for the pair sum | 400 |
| `FlingStress` | scale normalised pivot stress | 500 |
| `FlingVis` | neighbour embedding plus node edge clearance and crossings | 2000 |
| `FlingBlend` | a weighted blend of two energies, conditioned on the weight | 1500 |

### Parameters

Shared by all variants.

| name | default | meaning |
|---|---|---|
| `n_components` | 3 | output dimension, use 2 for a drawing |
| `seed` | 0 | random seed |
| `iters` | per variant | optimiser steps |
| `lr` | 0.005 | learning rate |
| `n_landmarks` | 64 | landmarks the diffusion potentials are measured to |
| `alpha` | 0.05 | teleport of the diffusion, which sets how far a node sees |
| `diff_K` | 20 | diffusion steps behind each potential |
| `width` | 128 | hidden width of the coordinate network |
| `depth` | 2 | hidden layers |
| `w0` | 4.0 for `Fling`, 2.0 otherwise | activation frequency, used only by siren, finer and gabor |
| `act` | `gelu` | activation |

Specific to `FlingStress`.

| name | default | meaning |
|---|---|---|
| `npiv` | `None` | pivot columns, `None` picks 400 or fewer on small graphs |

Specific to `FlingVis`.

| name | default | meaning |
|---|---|---|
| `w_ne` | 0.3 | weight of the node edge clearance term |
| `w_cross` | 0.3 | weight of the crossing term |
| `clear` | 0.6 | clearance target as a fraction of the median edge length |
| `n_neg` | 4096 | sampled negatives per step |
| `exaggeration` | 4.0 | attraction factor over the first third of training |

Specific to `FlingBlend`.

| name | default | meaning |
|---|---|---|
| `lam` | `None` | pin to one weight, `None` trains the whole family |
| `n_lam` | 4 | weights drawn per step, so a step costs `n_lam` objective evaluations |

`alpha` sets how far a node sees. Each potential is a personalised PageRank whose step `k` carries
weight `alpha * (1 - alpha) ** k`, so the mass is spent after roughly `1 / alpha` hops and `diff_K`
only has to be long enough to reach that. Raising `diff_K` alone does nothing, because the weights
have already decayed.

The default sees about twenty hops. Leave it there unless a drawing looks wrong, because the effect
of changing it is large, not monotone, and different on every graph. Measured on sampled stress, a
120 by 120 grid folds at the default and wants a lower `alpha` (0.046 down to 0.012 at 0.01), a 400
by 400 grid draws a clean lattice at the default and gets worse at 0.02 (0.012 up to 0.061), and the
million node road network is fine at the default but collapses onto a line at 0.005 (0.015 up to
0.274). Sweep it over several seeds and look at the drawing rather than the number.

### Baselines and metrics

`fling.baselines` holds `sgd_stress`, `fa2`, `tsnet`, `tsnet_star`, `NNPNet`, `impred` and the
landmark readouts. `fling.metrics` holds `evaluate`, `normalized_stress`,
`neighborhood_preservation` and `fast_D`. `fling.data` loads the benchmark graphs.

## Scripts

Each script computes its own data, caches it under `cache`, and writes to `tables` and `figures`.
Running one twice is a no op. Defaults reproduce the paper.

| script | output |
|---|---|
| `model_comparison.py` | four metric tables, the rank table, both gallery figures |
| `lean_tables.py` | `tab_main`, `tab_lean`, `tab_lean_controls` |
| `m_sweep.py` | `m_sweep.pdf` |
| `lambda_family.py` | `fam_family.pdf` |
| `tradeoff.py` | `fig_a3_tradeoff.pdf` |
| `scaling.py` | `exp_scaling_2x2.pdf` |
| `architecture.tex` | `architecture.pdf`, built with pdflatex |
| `million.py` | times Fling against s_gd2 on a graph of about a million nodes |
| `draw_large.py` | rasterises those layouts with datashader |

`lib.py` is not a script and writes nothing. It holds the sample splitting and held out stress
helpers that `lean_tables.py` and `m_sweep.py` both need, which were duplicated in the two.

```
python scripts/model_comparison.py
python scripts/scaling.py --seeds 0
python scripts/million.py --graph roadnet
python scripts/draw_large.py --graph roadnet
```

`million.py` takes `--graph roadnet` for the Pennsylvania road network and `--graph delaunay
--n 1000000` for a triangulation of random points. Matplotlib cannot draw a million nodes, so
`draw_large.py` uses datashader, which rasterises the edges into a raster of counts instead of
drawing each one. It needs `pip install datashader`.

The model comparison and the scaling study are the expensive ones. Scaling runs to a million nodes
and gates a method out of larger sizes once it passes a twenty minute cap, so a line that stops early
in the figure was gated rather than failed.

## Data

The benchmark graphs live in `data`. Files for 3elt, jagmesh8, dwt_1005, cora and ego-Facebook are
read from there. The rest are generated by networkx.

`roadNet-PA.txt.gz` is the Pennsylvania road network from SNAP, 1,087,562 nodes and 1,541,514 edges
in its largest component.

```
curl -o data/roadNet-PA.txt.gz https://snap.stanford.edu/data/roadNet-PA.txt.gz
```
