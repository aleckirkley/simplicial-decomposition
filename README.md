# Simplicial decomposition of hypergraphs

Code for decomposing a hypergraph into a set of latent simplices, whose
observed faces are better described as parts of a single group interaction,
and a residual set of independent hyperedges. The decomposition minimizes a
description length derived from a Bayesian generative model (equivalently,
it is a maximum a posteriori estimate) and has no free parameters.

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
simplicial structure was found and every hyperedge is returned as residual.
The null model is a uniformly random hypergraph with the same number of
hyperedges of each size, so random hypergraphs give `eta = 1`.

Since the null model treats all nodes alike, very heterogeneous degrees can
give some compression on their own (e.g. a large simplex made of high-degree
nodes). To check for this, compare `eta` with the value obtained on a
degree-preserving randomization of the data, for instance one made by
shuffling node memberships across hyperedges while keeping hyperedge sizes
fixed.

## Large hypergraphs

Two limits keep the run time manageable on large data: the number of
candidate pairs considered per node (`SD_NODE_PAIR_CAP`, default 1000) and
the number of new candidates considered after each merge (`SD_HUB_CAP`,
default 100). Both can be set as environment variables. They have no effect
on small and medium-sized hypergraphs, and with them we have run the method
on hypergraphs with close to a million hyperedges. Since every merge is
scored by its exact change in description length, the limits can only cause
a merge to be missed, never a bad merge to be made. Results are
deterministic.
