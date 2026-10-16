# Simplicial decomposition of hypergraphs

Parameter-free decomposition of a hypergraph into its **simplicial
component** — a set of latent simplices whose observed faces are more
parsimoniously described as fragments of coherent group interactions —
and a **residual** of independent hyperedges. The method minimizes a
description length derived from a Bayesian generative model (equivalently,
performs maximum-a-posteriori inference) and has no tunable parameters.

## Contents

- `simplicial_decomposition.py` — the complete algorithm in one file,
  extracted verbatim from the code used for every result in the paper.
  It needs only the Python standard library (Python >= 3.9).
- `reproduce_fig1.py` — reproduces the paper's Fig. 1 example:
  decomposes a small synthetic hypergraph and prints the observed
  hypergraph, the inferred simplices with their hosted hyperedges, the
  residual, and the compression ratio.

## Quick start

```python
from simplicial_decomposition import decompose

hyperedges = [
    {1, 2, 3, 4},          # any iterables of hashable node labels
    {1, 2, 3}, {2, 3, 4},
    {1, 2}, {3, 4},
    {5, 6},
]
result = decompose(hyperedges)

result['S']     # inferred simplices (list of frozensets of nodes)
result['H']     # residual hyperedges
result['F']     # simplex -> hyperedges it hosts
result['eta']   # compression ratio L / L0 (1.0 = no simplicial structure)
```

To reproduce Fig. 1:

```
python reproduce_fig1.py
```

## What the output means

Each inferred simplex `s` is a claim that the hyperedges nested inside
it are better explained as noisy observations of one latent group
interaction on `s`'s nodes than as independent interactions. The
compression ratio `eta` in (0, 1] summarizes how much simplicial
structure the data contain (lower = more); `eta = 1` means the method
found no evidence of simplicial structure and returned the null
decomposition. The null description is the code of a uniformly random
hypergraph with the observed numbers of hyperedges of each size, so
random hypergraphs give `eta = 1`: the method abstains rather than
hallucinating structure on random data.

Because the null is uniform over nodes, strongly heterogeneous node
degrees alone can produce mild compression (for example, a large
"simplex" of high-degree nodes). To separate simplicial structure from
degree effects, compare `eta` with its value on a degree-preserving
randomization of your data, e.g. one obtained by shuffling node
incidences across hyperedges while keeping hyperedge sizes fixed.

## Scaling

The file includes the paper's two resource budgets: a per-node
candidate-pair budget (`SD_NODE_PAIR_CAP`, default 1000) and a
per-object re-pairing budget (`SD_HUB_CAP`, default 100), both
settable as environment variables. Neither engages on small or moderate
data, and with them the method has been run on hypergraphs with close to
10^6 hyperedges. Every move is scored by its exact change in description
length, so the budgets can only omit candidate merges and never apply an
unsound one. Results are deterministic.

