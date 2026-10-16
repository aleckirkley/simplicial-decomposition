# Simplicial decomposition of hypergraphs

Code for decomposing a hypergraph into a set of latent simplices, whose
observed faces are better described as parts of a single group interaction,
and a residual set of independent hyperedges. The decomposition minimizes a
description length derived from a Bayesian generative model and has no free parameters.

## Files

- `simplicial_decomposition.py`: the full method in a single file. It is
  the same code used for the results in the paper and only needs the Python
  standard library (Python 3.9 or later).
- `reproduce_fig1.py`: runs the example from Fig. 1 of the paper and prints
  the hypergraph, the inferred simplices with the hyperedges they host, the
  residual hyperedges, and the compression ratio.

## Usage

```python
from simplicial_decomposition import decompose

hyperedges = [
    {1, 2, 3, 4},
    {1, 2, 3}, {2, 3, 4},
    {1, 2}, {3, 4},
    {5, 6},
]
result = decompose(hyperedges)

result['S']     # inferred simplices (frozensets of nodes)
result['H']     # residual hyperedges
result['F']     # hyperedges hosted by each simplex
result['eta']   # compression ratio L / L0
```

Node labels can be any hashable objects. To reproduce Fig. 1, run

```
python reproduce_fig1.py
```

## Interpreting the output

Each simplex `s` in the output means that the hyperedges inside `s` are
better explained as noisy observations of one group interaction on the nodes
of `s` than as separate interactions. The compression ratio `eta` lies in
(0, 1], with lower values meaning more simplicial structure. If `eta = 1`, no
simplicial structure was found and every hyperedge is returned as independent.
The null model is a uniformly random hypergraph with the same number of
hyperedges of each size, so random hypergraphs give `eta = 1`.

## Large hypergraphs

The number of candidate pairs considered per node (`SD_NODE_PAIR_CAP`, default 1000) and
the number of new candidates considered after each merge (`SD_HUB_CAP`,
default 100) keep the run time manageable on large data. Both can be set as environment variables, and have no effect
on small and medium-sized hypergraphs. 
