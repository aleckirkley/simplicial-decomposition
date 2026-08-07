"""
Reproduce the Fig.-1 example of the paper: decompose a small deterministic
hypergraph (two noisy simplices, one complete simplex, and seven sparse
hyperedges) and print the observed hypergraph and its inferred simplicial
decomposition.

Usage:  python reproduce_fig1.py
Dependencies: numpy (via simplicial_decomposition).
"""
from simplicial_decomposition import decompose

# ── the deterministic Fig.-1 hypergraph ──────────────────────────────────────
# A 5-node and a 4-node simplex with roughly half of their proper faces
# removed, a complete 3-node simplex, and seven sparse hyperedges (two
# connectors, five danglers) tying everything into one component.
G = [
    # S1 = {0..4}: top face + 2 quadruples + 3 triples + 5 pairs
    {0, 1, 2, 3, 4},
    {0, 1, 2, 3}, {1, 2, 3, 4},
    {0, 1, 2}, {2, 3, 4}, {0, 3, 4},
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 4},
    # S2 = {5..8}: top face + 2 triples + 3 pairs
    {5, 6, 7, 8},
    {5, 6, 7}, {6, 7, 8},
    {5, 6}, {7, 8}, {5, 8},
    # S3 = {9,10,11}: complete triangle
    {9, 10, 11},
    {9, 10}, {9, 11}, {10, 11},
    # sparse connectors / danglers (pairwise node-disjoint)
    {3, 8}, {7, 9, 12}, {0, 13}, {2, 14, 15}, {6, 16}, {10, 17, 18}, {11, 19},
]


def fmt(e):
    return '{' + ','.join(str(v) for v in sorted(e)) + '}'


def fmt_row(edges, indent='  ', width=76):
    """Wrap a list of hyperedges into tidy indented lines."""
    lines, line = [], indent
    for e in edges:
        tok = fmt(e) + ' '
        if len(line) + len(tok) > width and line.strip():
            lines.append(line.rstrip())
            line = indent
        line += tok
    lines.append(line.rstrip())
    return '\n'.join(lines)


# ── observed hypergraph ──────────────────────────────────────────────────────
nodes = sorted(set(v for e in G for v in e))
print(f'observed hypergraph G: {len(G)} hyperedges on {len(nodes)} nodes')
print(fmt_row(sorted((frozenset(e) for e in G),
                     key=lambda e: (-len(e), tuple(sorted(e))))))

# ── decomposition ────────────────────────────────────────────────────────────
result = decompose(G)

print('\ninferred decomposition:')
for s in result['S']:
    faces = result['F'][s]
    print(f'\n  simplex {fmt(s)}  --  hosts {len(faces)} observed hyperedges:')
    print(fmt_row(faces, indent='    '))

print(f'\n  residual H: {len(result["H"])} independent hyperedges:')
print(fmt_row(result['H'], indent='    '))

print(f'\ndescription length L = {result["L"]:.1f} bits '
      f'(null L0 = {result["L0"]:.1f} bits)')
print(f'compression ratio eta = L/L0 = {result["eta"]:.3f}')
