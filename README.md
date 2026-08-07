# Simplicial decomposition of hypergraphs

Parameter-free decomposition of a hypergraph into its **simplicial
component** — a set of latent simplices whose observed faces are more
parsimoniously described as fragments of coherent group interactions —
and a **residual** of independent hyperedges. The method minimizes a
description length derived from a Bayesian generative model.

## Contents

- `simplicial_decomposition.py` — the complete algorithm for running the decomposition.
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
structure the data contain (lower = more).

## Scaling

The file includes the paper's three resource budgets (a per-node
candidate-pair budget, a per-object candidate budget, and a memory
bound under which candidate seeding switches to a MinHash-LSH
generator, implemented in numpy), allowing for easy scalability to very large systems 

