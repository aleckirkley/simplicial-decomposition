"""
simplicial_decomposition.py -- MDL simplicial decomposition of hypergraphs.

Decomposes an observed hypergraph G into a set S of latent simplices and
a residual set H of independent hyperedges by minimizing the description
length (equivalently, maximizing the Bayesian posterior of the paper's
generative model):

    L(S,H) = sum_{s in S} [ n(s) + log2 C(N, n(s)) ]                 (simplices)
           + sum_{s in S} sum_{e in F(s)} log2 C(n(s), n(e))         (hosted faces)
           + sum_{h in H} log2 C(N, n(h)),                           (residual)

where n(.) counts nodes, N is the number of nodes in the system, and
F(s) is the set of observed hyperedges hosted by simplex s (each
hyperedge is hosted by its smallest enclosing simplex; a simplex's own
top face, when observed, is hosted at zero cost). The decomposition is
found with the two-phase greedy algorithm of the paper: a bottom-up
collapse of the nestedness DAG (Phase 1) followed by best-first merging
of overlapping objects with exact description-length deltas (Phase 2).

This file is extracted verbatim from the code used to produce every
result in the paper (only plotting, corpus-management, and unit-test
machinery are omitted), so it reproduces the paper's outputs exactly,
including the three resource budgets used on very large corpora (a
per-node candidate-pair budget, a per-object candidate budget, and a
memory bound under which candidate seeding switches to a MinHash-LSH
generator; none of them engage on small or moderate data).

Quick start:

    from simplicial_decomposition import decompose
    result = decompose(hyperedges)          # iterable of node collections
    result['S']      # inferred simplices (list of frozensets)
    result['H']      # residual hyperedges (list of frozensets)
    result['eta']    # compression ratio L / L0 (1.0 = no simplicial structure)

Dependencies: numpy. (Python >= 3.9.)

Reference: A. Kirkley and F. Battiston, "Finding noisy simplices in
complex systems" (2026). Please cite the paper if you use this code.
"""

import heapq
import math
import os
import random
import time
import zlib
from collections import Counter, defaultdict

import numpy as np


def loggamma(x):
    """Natural log-gamma (stdlib; scipy-free)."""
    return math.lgamma(x)


SIZE_PRIOR = 'geometric'


NODE_PAIR_CAP = int(float(os.environ.get('SD_NODE_PAIR_CAP', 1_000)))


HUB_CAP = int(float(os.environ.get('SD_HUB_CAP', 100)))


SEED_MEM_CAP = int(float(os.environ.get('SD_SEED_MEM_CAP', 6e8)))


PHASE2_HUB_TAU = 64


LSH_NUM_PERM = int(float(os.environ.get('SD_LSH_NUM_PERM', 25)))


LSH_THRESHOLD = float(os.environ.get('SD_LSH_THRESHOLD', 0.5))


last_run_info = {}


def _size_prior_on():
    return SIZE_PRIOR == 'geometric'


def logchoose(n,k):
    """
    log binomial coefficient
    """
    return loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)


def _logchoose(n, k):
    """log2 C(n, k). Cached: (n, k) pairs repeat heavily across edges of the
    same size, so memoizing avoids recomputing loggamma calls."""
    if k < 0 or k > n:
        return float('inf')
    return logchoose(n, k) / math.log(2)


def _enclosing_key(s):
    """Canonical tie-break key for selecting the smallest enclosing simplex:
    size first, then lexicographic on the sorted node tuple. Guarantees a
    deterministic choice whenever multiple candidates in S have equal size
    (ties do not affect L -- see SI Eq. assignment -- but an implementation
    must still commit to *some* fixed rule for reproducibility)."""
    return (len(s), tuple(sorted(s)))


def _assign_faces(G_frozen, S):
    """Assign each edge in G_frozen to its smallest enclosing simplex in S
    (ties broken via _enclosing_key), matching the SI's assignment rule.
    Returns (faces_assigned, covered) with the same contents as the naive

        faces_assigned = {s: [] for s in S}
        covered = set()
        for e in G_frozen:
            covering = [s for s in S if e <= s]
            if covering:
                best = min(covering, key=_enclosing_key)
                faces_assigned[best].append(e)
                covered.add(e)

    but found via a node -> simplices-of-S index, so each edge only checks
    the (small) set of simplices incident to *every* one of its nodes --
    the true candidates for e <= s -- instead of scanning all of S. This
    turns an O(E*|S|) full scan into O(E * degree-in-S), which matters
    once |S| is large (e.g. right after phase 1, before phase 2 has had a
    chance to shrink it, or for corpora with tens of thousands of edges).
    """
    faces_assigned = {s: [] for s in S}
    covered = set()
    if not S:
        return faces_assigned, covered

    node_index = defaultdict(set)
    for s in S:
        for v in s:
            node_index[v].add(s)

    for e in G_frozen:
        if not e:
            covering = list(S)
        else:
            it = iter(e)
            candidates = set(node_index.get(next(it), ()))
            for v in it:
                candidates &= node_index.get(v, set())
            covering = [s for s in candidates if e <= s]
        if covering:
            best = min(covering, key=_enclosing_key)
            faces_assigned[best].append(e)
            covered.add(e)
    return faces_assigned, covered


def description_length(G_frozen, S, N):
    """L(S, H) -- Eq. (L): node-set + face-encoding cost for simplices in S,
    plus isolation cost for the residual H = G \\ (faces covered by S).

    Each G-face is assigned to its smallest enclosing simplex (ties broken
    deterministically via _enclosing_key), matching the SI's assignment
    rule (Eq. assignment).
    """
    S = list(S)
    faces_assigned, covered = _assign_faces(G_frozen, S)
    H = G_frozen - covered
    dl = (sum(_logchoose(N, len(h)) for h in H)
          + sum(_logchoose(N, len(s)) + sum(_logchoose(len(s), len(e)) for e in faces_assigned[s])
                for s in S))
    if _size_prior_on():
        # geometric size prior: n(s) bits per non-singleton simplex.
        # (Hyperedge sizes cancel against the null and are never added;
        # singletons are charged as isolated edges, consistent with the
        # greedy's internal accounting and with _finalise moving them to
        # H -- under this prior that move is strictly optimal.)
        dl += sum(len(s) for s in S if set(faces_assigned[s]) != {s})
    return dl


def baseline_DL(G_frozen, N):
    """L0: description length when every edge is isolated (no simplices,
    S = empty set). This is the null model compared against throughout."""
    return sum(_logchoose(N, len(e)) for e in G_frozen)


def _finalise(G_frozen, S, DL, L0):
    """Drop neutral singletons (A(s) = {s}: costs exactly the same as
    leaving s in H, so they represent no real compression), then return
    (S, H, DL, eta). Uses the same lexicographic tie-break as
    description_length so the returned partition is deterministic."""
    faces_assigned, _ = _assign_faces(G_frozen, S)
    S = [s for s in S if set(faces_assigned[s]) != {s}]
    _, covered = _assign_faces(G_frozen, S)
    H = G_frozen - covered
    return S, H, DL, DL / L0 if L0 > 0 else float('nan')


def _build_nestedness_dag(G_list):
    """Return child -> set-of-direct-parents in the nestedness partial order
    (h1 -> h2 when h1 is a proper subset of h2, with no intermediate
    hyperedge in G).

    h2 can only be a superset of h1 if h2 is incident to *every* node of
    h1, so the candidate supersets of h1 are exactly the intersection of
    the incident-edge sets of h1's nodes (via a node -> incident-edges
    index built once, up front). This replaces an O(E^2) all-pairs subset
    scan with a cost proportional to node degrees, which is what actually
    matters on sparse empirical hypergraphs where E can be in the
    millions but per-node degree is orders of magnitude smaller. The
    final Hasse reduction (dropping non-direct parents) is unchanged and
    stays cheap since each hyperedge's candidate-parent set is now small.
    """
    node_index = defaultdict(set)
    for h in G_list:
        for v in h:
            node_index[v].add(h)

    parents = {}
    for h1 in G_list:
        if not h1:
            parents[h1] = set()
            continue
        it = iter(h1)
        candidates = set(node_index[next(it)])
        for v in it:
            candidates &= node_index[v]
        candidates.discard(h1)
        parents[h1] = {h2 for h2 in candidates if h1 < h2}

    # Hasse reduction: keep only the minimal parents under the subset order.
    # Processing candidates in ascending size and testing each only against
    # the already-kept minimal set is exact (if p is dominated by some
    # non-minimal q < p, transitivity gives a kept r < q < p that also
    # dominates p) and costs O(|pars| * |direct|) per child instead of the
    # previous all-pairs O(|pars|^2) -- the difference matters on heavily
    # nested corpora where a small hyperedge can have hundreds of parents
    # but only a handful of direct ones.
    direct = {}
    for child, pars in parents.items():
        kept = []
        for p in sorted(pars, key=_enclosing_key):
            if not any(q < p for q in kept):
                kept.append(p)
        direct[child] = set(kept)
    return direct


def phase1(G_frozen, N, logfile=None):
    """Bottom-up DAG collapse (Eq. delta-dag): absorb each edge into
    whichever direct parent yields the most-negative re-encoding gain.

    Parents are iterated in a fixed deterministic order (size, then sorted
    node tuple), and all viable parents are evaluated before committing to
    the one with the smallest delta, so the result is independent of
    Python's set/dict iteration order.
    """
    G_list = list(G_frozen)
    sfm = {h: frozenset({h}) for h in G_list}
    dag = _build_nestedness_dag(G_list)
    # Per-simplex face-SIZE histograms: the absorption delta only depends on
    # the sizes of the already-absorbed faces, so summing over the histogram
    # (a handful of distinct sizes) instead of over every face is exact and
    # turns the per-parent evaluation cost from O(|A(h)|) into O(k).
    # Identity used (equivalent to the original explicit sum):
    #   delta = sum_j hist_h[j] * [logC(n_par, j) - logC(n_h, j)]
    #           - logC(N, n_h)
    # (the e = h term of the histogram sum contributes exactly the original
    # leading +logC(n_par, n_h), since logC(n_h, n_h) = 0).
    hist = {h: Counter((len(h),)) for h in G_list}
    _sp = _size_prior_on()
    for h in sorted(G_list, key=_enclosing_key):
        if h not in sfm:
            continue
        Gh = sfm[h]
        n_h = len(h)
        hist_h = hist[h]
        base = -_logchoose(N, n_h)
        if _sp and sum(hist_h.values()) > 1:
            # geometric size prior: h's own (non-singleton) simplex, and
            # its n(h)-bit size cost, disappear on absorption
            base -= n_h
        self_costs = {j: _logchoose(n_h, j) for j in hist_h}
        best_delta, best_hpar = 0.0, None
        for hpar in sorted(dag.get(h, set()), key=_enclosing_key):
            if hpar not in sfm:
                continue
            n_p = len(hpar)
            delta = base + sum(c * (_logchoose(n_p, j) - self_costs[j])
                               for j, c in hist_h.items())
            if _sp and sum(hist[hpar].values()) == 1:
                # the parent was a singleton and becomes a real simplex,
                # incurring its n(p)-bit size cost
                delta += n_p
            if logfile:
                logfile.write(f"Phase1: {sorted(h)} -> {sorted(hpar)}  delta={delta:.3f}\n")
            if delta < best_delta:
                best_delta, best_hpar = delta, hpar
        if best_hpar is not None:
            sfm[best_hpar] = sfm.get(best_hpar, frozenset()) | Gh
            hist[best_hpar] = hist.get(best_hpar, Counter()) + hist_h
            del sfm[h]
    return sfm


def phase2(simplex_faces_map, G_frozen, N, logfile=None,
           approx=False, hub_cap=None, size_cap=None,
           lsh_num_perm=None, lsh_threshold=None,
           stochastic_eps=0.0, theta=0.0, budget_s=None,
           node_pair_cap=None):
    """Intersection-graph greedy merges via min-heap (SI "Phase 2").

    hub_cap / lsh_num_perm / lsh_threshold left at None resolve to the
    module-level resource constants, so direct callers get the same
    configuration as greedy_optimizer.

    2026-07-18 rework (exact -- same merge criterion and same outputs as
    the direct implementation, up to floating-point summation order):
    the per-candidate-pair merge delta used to be recomputed from scratch
    by enumerating every hyperedge nested in the pair's union, which costs
    O(|G|) per evaluation once a growing simplex covers a large fraction
    of the nodes; profiling showed >95% of wall-clock time there on
    corpora that grow giant simplices. The rework maintains, per current
    simplex s:

      nested[s]   every G-edge contained in s (complete by construction:
                  for a merge-born union u = s1 | s2, an edge e <= u
                  satisfies e <= s1, e <= s2, or e meets both s1\s2 and
                  s2\s1 -- an exact decomposition -- so
                  nested[u] = nested[s1] | nested[s2] | cross(s1,s2),
                  with cross() enumerable from the incident edges of the
                  SMALLER node-difference only)
      owned[s]    A(s) as before (phase-1 absorptions; for merge-born
                  simplices owned == nested, matching the previous
                  behavior of assigning a union every edge nested in it)
      size histograms and three cached cost sums per simplex
                  (sum logC(N, n(e)) over nested and over owned, and
                  sum logC(n(s), n(e)) over owned)

    from which the exact pairwise delta of Eq. (delta-ss) -- including the
    documented "spanning" approximation, unchanged -- is assembled in
    O(k + |shared| + |cross|) time, where shared/cross are enumerated via
    the smaller of the two operands. Set unions steal the larger operand
    (union-find style), so the total maintenance cost over a sequence of
    merges is near-linear rather than quadratic in the giant simplex's
    face count. Heap bookkeeping, push order, and tie handling are kept
    byte-identical to the previous implementation, so the only possible
    output differences are floating-point summation-order effects on
    exactly tied merge deltas (rare, symmetric-structure cases; verified
    at ~1 in 50 synthetic stress instances, with either choice a valid
    greedy step).
    """
    if hub_cap is None:
        hub_cap = HUB_CAP
    if lsh_num_perm is None:
        lsh_num_perm = LSH_NUM_PERM
    if lsh_threshold is None:
        lsh_threshold = LSH_THRESHOLD
    heap = []
    valid = {}                          # pair-key (frozenset) -> delta
    s_pairs = {}                        # simplex -> set of pair-keys it appears in

    edge_index = defaultdict(set)       # node -> incident edges of G_frozen (fixed)
    for e in G_frozen:
        for v in e:
            edge_index[v].add(e)
    deg = {v: len(es) for v, es in edge_index.items()}
    kmax = max((len(e) for e in G_frozen), default=0)
    # per-node allowance for the cross-edge charge: any cross edge of the
    # pair (s1, s2) is incident to the smaller node-difference, and each
    # contributes at most logC(N, n(e)) to the delta's spanning charge, so
    # sum_{v in diff} nodeCrossN[v] soundly upper-bounds the total cross
    # contribution (used to prune pairs before enumerating cross edges).
    nodeCrossN = {v: sum(_logchoose(N, len(e)) for e in es)
                  for v, es in edge_index.items()}

    # ── per-simplex incremental state ────────────────────────────────────────
    sfm = {}          # simplex -> owned (A(s)) as a mutable set
    nested = {}       # simplex -> all G-edges contained in s (mutable set)
    histN = {}        # simplex -> Counter of nested face sizes
    nestN_c = {}      # simplex -> sum logC(N, n(e)) over nested
    ownN_c = {}       # simplex -> sum logC(N, n(e)) over owned
    ownSelf_c = {}    # simplex -> sum logC(n(s), n(e)) over owned
    degsum = {}       # simplex -> sum of node degrees (for the cross bound)
    sxc = {}          # simplex -> sum of nodeCrossN over the simplex's nodes

    # node -> live simplices index, built first so the nested-set
    # initialization below can use the inverted rarest-node scan. Kept
    # incrementally maintained through merges (see the merge block).
    simplex_index = defaultdict(set)
    for s in simplex_faces_map:
        for v in s:
            simplex_index[v].add(s)

    for s, faces in simplex_faces_map.items():
        sfm[s] = set(faces)
        nested[s] = set()
        s_pairs[s] = set()

    # ── nested-set initialization, inverted (2026-07-18, exact) ──────────
    # The previous loop scanned, for every simplex, every incident edge of
    # every member node: sum_s sum_{v in s} deg_G(v), i.e. O(d^2) for a hub
    # of degree d. Inverting the loop over EDGES and scanning only each
    # edge's rarest node's simplex-index entry gives the identical nested
    # sets at cost sum_e min-simplex-degree(e): any simplex s containing e
    # contains every node of e, so it necessarily appears in
    # simplex_index[v] for e's least-frequent node v.
    for e in G_frozen:
        v_star = min(e, key=lambda v: len(simplex_index.get(v, ())))
        for t in simplex_index.get(v_star, ()):
            if e <= t:
                nested[t].add(e)

    for s in simplex_faces_map:
        own = sfm[s]
        histN[s] = Counter(len(e) for e in nested[s])
        n_s = len(s)
        nestN_c[s] = sum(c * _logchoose(N, j) for j, c in histN[s].items())
        hA = Counter(len(e) for e in own)
        ownN_c[s] = sum(c * _logchoose(N, j) for j, c in hA.items())
        ownSelf_c[s] = sum(c * _logchoose(n_s, j) for j, c in hA.items())
        degsum[s] = sum(deg.get(v, 0) for v in s)
        sxc[s] = sum(nodeCrossN.get(v, 0.0) for v in s)

    # ── phase-2 entry invariant (2026-07-28 bugfix) ──────────────────────
    # The merge-delta bookkeeping below charges every nested face that is
    # NOT owned by an operand at its isolated cost logC(N, n(e)). That is
    # EXACT when the face's current owner is a singleton object (whose
    # entire cost is that isolated cost) but WRONG when the owner is a
    # real multi-face simplex nested inside a live object: the face is
    # cheaply hosted there, so the "isolated" credit is spurious, and on
    # states with real nested simplices phase 2 accepted merges whose true
    # description-length change was strongly positive (observed: L=7
    # reconstruction sweeps under the geometric size prior, where phase 1
    # correctly declines marginal absorptions and thereby leaves real
    # nested simplices alive -- the legacy prior left almost none, which
    # is why this stayed latent). Restore the invariant the formulas
    # assume: every real simplex nested inside another live object is
    # absorbed into its SMALLEST enclosing live object up front,
    # accepting the (typically marginal) cost; singletons stay put, since
    # their accounting is exact. Ascending size order cascades chains.
    # The post-merge nested-simplex cleanup maintains the invariant for
    # the rest of the run.
    _forced = 0
    for s in sorted(list(sfm), key=lambda t: (len(t), _enclosing_key(t))):
        if s not in sfm or len(sfm[s]) <= 1:
            continue
        v_star = min(s, key=lambda v: len(simplex_index.get(v, ())))
        hosts = [t for t in simplex_index.get(v_star, ())
                 if t is not s and s < t]
        if not hosts:
            continue
        t = min(hosts, key=lambda h: (len(h), _enclosing_key(h)))
        moved = sfm.pop(s)
        n_t = len(t)
        sfm[t] = set(sfm[t]) | moved
        hM = Counter(len(e) for e in moved)
        ownN_c[t] += sum(c * _logchoose(N, j) for j, c in hM.items())
        ownSelf_c[t] += sum(c * _logchoose(n_t, j) for j, c in hM.items())
        # nested[t] / histN[t] already contain these faces (G-edge based)
        for v in s:
            simplex_index[v].discard(s)
        for dd in (nested, histN, nestN_c, ownN_c, ownSelf_c, degsum, sxc):
            dd.pop(s, None)
        s_pairs.pop(s, None)
        _forced += 1
    if _forced and logfile:
        logfile.write(f"Phase2 entry: force-absorbed {_forced} real nested "
                      f"simplices (bookkeeping invariant)\n")

    # ── collective absorption pass (2026-07-29 v15) ──────────────────────
    # Under the geometric size prior a SINGLETON parent's first absorption
    # pays the full +n(p) size cost, so phase 1's one-at-a-time deltas
    # decline every child individually even when absorbing them
    # collectively is strongly favorable (house-bills: a 400-cosponsor
    # bill with hundreds of nested small bills -- each child saves a few
    # bits, the size cost is paid ONCE). Exact phase 2 recovers this via
    # (child, parent) merge pairs, but the LSH generator never proposes
    # them (Jaccard(child, big parent) ~ 0), which produced eta = 0.9994
    # on house-bills. Fix at the source: for each singleton parent p
    # (ascending size, so children go to their smallest enclosing live
    # parent, matching the assignment rule), every nested singleton child
    # has individually negative absorption delta once the size cost is
    # paid, so absorb ALL of them iff
    #   n(p)*[size prior] + sum_c [logC(n(p),n(c)) - logC(N,n(c))] < 0.
    # Exactly computable from the nested[] caches; a no-op under the
    # legacy prior (phase 1 already absorbed every such child).
    _coll = 0
    _sp_on = _size_prior_on()
    for p in sorted(list(sfm), key=lambda t: (len(t), _enclosing_key(t))):
        if p not in sfm or len(sfm[p]) != 1:
            continue
        # only inclusion-MAXIMAL live parents: absorbing into a nested
        # parent would recreate the real-nested configuration the
        # invariant pass above just eliminated; a non-maximal parent's
        # children (and the parent itself, as an edge) are collected by
        # its enclosing parent later in this ascending sweep.
        v_star = min(p, key=lambda v: len(simplex_index.get(v, ())))
        if any(t is not p and p < t for t in simplex_index.get(v_star, ())):
            continue
        kids = [e for e in nested[p]
                if e != p and e in sfm and len(sfm[e]) == 1]
        if not kids:
            continue
        n_p = len(p)
        total = (n_p if _sp_on else 0) + sum(
            _logchoose(n_p, len(e)) - _logchoose(N, len(e)) for e in kids)
        if total >= 0:
            continue
        moved = set(kids)
        hM = Counter(len(e) for e in moved)
        ownN_c[p] += sum(c * _logchoose(N, j) for j, c in hM.items())
        ownSelf_c[p] += sum(c * _logchoose(n_p, j) for j, c in hM.items())
        sfm[p] = set(sfm[p]) | moved
        for e in kids:
            for v in e:
                simplex_index[v].discard(e)
            for dd in (sfm, nested, histN, nestN_c, ownN_c, ownSelf_c,
                       degsum, sxc):
                dd.pop(e, None)
            s_pairs.pop(e, None)
        _coll += len(kids)
    if _coll and logfile:
        logfile.write(f"Phase2 entry: collectively absorbed {_coll} nested "
                      f"singletons into their maximal singleton parents\n")

    def _candidates(node_set):
        cands = set()
        for v in node_set:
            cands |= simplex_index.get(v, set())
        return cands

    # Per-simplex overlap-neighbor sets, maintained incrementally: a merge's
    # union overlaps exactly the (live) simplices that overlapped either
    # operand, so nbrs[union] = nbrs[s1] | nbrs[s2] (steal the larger),
    # filtered for liveness lazily at use. This replaces re-scanning the
    # node -> simplices index over every node of the union after each merge,
    # which is O(sum of the union's node degrees) and dominates once a
    # growing simplex covers most of the graph.
    # ── heavy/light neighbor split (2026-07-18) ──────────────────────────
    # Storing every overlap-neighbor materializes the hub pair explosion
    # (a node in d survivors contributes d(d-1) stored entries) before a
    # single merge happens. Instead: nbrs[s] stores only neighbors
    # reachable through LIGHT nodes (simplex-degree <= PHASE2_HUB_TAU at
    # initialization); neighbors through the few HEAVY nodes are resolved
    # on demand from simplex_index, which is now maintained incrementally
    # through merges and therefore always current. Coverage is exact: any
    # overlapping pair shares at least one node, found either in a stored
    # light entry or in a heavy node's live index entry.
    heavy_nodes = {v for v, ts in simplex_index.items()
                   if len(ts) > PHASE2_HUB_TAU}
    hnodes = {s: frozenset(v for v in s if v in heavy_nodes) for s in sfm}
    nbrs = {s: set() for s in sfm}
    for v, ts in simplex_index.items():
        if v in heavy_nodes or len(ts) < 2:
            continue
        tl = list(ts)
        for a in tl:
            nb = nbrs[a]
            for b in tl:
                if b is not a:
                    nb.add(b)

    # Cached canonical sort key per simplex (the previous code recomputed
    # tuple(sorted(s)) inside every sorted() call -- O(n log n) per
    # comparison key, painful for giant simplices).
    _key_cache = {}
    def _ckey(s):
        k = _key_cache.get(s)
        if k is None:
            k = _enclosing_key(s)
            _key_cache[s] = k
        return k

    _sp = _size_prior_on()

    def _sp_term(s1, s2, u):
        """Geometric-size-prior contribution to the merge delta: the union
        always pays its n(u)-bit size cost; each operand's size cost is
        recovered only if it was a real (non-singleton) simplex."""
        return (u - (len(s1) if len(sfm[s1]) > 1 else 0)
                  - (len(s2) if len(sfm[s2]) > 1 else 0))

    def _l1_seed_excluded(a, b):
        """Single-overlap candidate restriction (SM Sec. I), applied
        UNIFORMLY at seeding and at post-merge re-pairing: for |a & b| == 1
        the exactly computable part of the delta (everything except the
        cross-edge credit) decides candidacy; returns True when the pair
        is excluded. Uses only per-simplex scalars, O(k)."""
        n1, n2 = len(a), len(b)
        u = n1 + n2 - 1
        C0 = _logchoose(N, u) - _logchoose(N, n1) - _logchoose(N, n2)
        U1 = sum(c * _logchoose(u, j) for j, c in histN[a].items())
        U2 = sum(c * _logchoose(u, j) for j, c in histN[b].items())
        excl = (C0 + U1 + U2
                - ownSelf_c[a] - ownSelf_c[b]
                - (nestN_c[a] - ownN_c[a]) - (nestN_c[b] - ownN_c[b]))
        if _sp:
            excl += _sp_term(a, b, u)
        return excl >= 0

    def _cross_terms(s1, s2, len_d1, len_d2):
        """Edges contained in s1 | s2 but nested in neither operand (they
        necessarily meet both node-differences, so enumerating candidates
        from EITHER difference side yields the same set exactly; the side
        with fewer nodes is chosen purely for speed, and only that side's
        node list is ever materialized -- building the larger side's list,
        or degree sums over it, dominated profiles when one operand covers
        most of the graph). Containment in the union is tested against the
        two operand node-sets directly, avoiding materializing the
        (possibly very large) union set on every candidate evaluation."""
        if len_d1 <= len_d2:
            diff = [v for v in s1 if v not in s2]
        else:
            diff = [v for v in s2 if v not in s1]
        n1_set, n2_set = nested[s1], nested[s2]
        seen = set()
        out = []
        for v in diff:
            for e in edge_index.get(v, ()):
                if e in seen:
                    continue
                seen.add(e)
                if e in n1_set or e in n2_set:
                    continue
                for w in e:
                    if w not in s1 and w not in s2:
                        break
                else:
                    out.append(e)
        return out

    def _delta(s1, s2):
        """Exact merge delta (Eq. delta-ss, same semantics as before)."""
        n1, n2 = len(s1), len(s2)
        small, big = (s1, s2) if n1 <= n2 else (s2, s1)
        inter = [v for v in small if v in big]
        u = n1 + n2 - len(inter)
        C0 = _logchoose(N, u) - _logchoose(N, n1) - _logchoose(N, n2)
        U1 = sum(c * _logchoose(u, j) for j, c in histN[s1].items())
        U2 = sum(c * _logchoose(u, j) for j, c in histN[s2].items())

        # ── exact evaluation ──────────────────────────────────────────────
        nsm, nbg = (nested[s1], nested[s2])
        if len(nsm) > len(nbg):
            nsm, nbg = nbg, nsm
        own1, own2 = sfm[s1], sfm[s2]
        sharedU = sharedN = sharedA_N = 0.0
        # |s1 & s2| <= 1: no hyperedge (>= 2 nodes) fits inside the
        # intersection, so the shared-face terms vanish identically and the
        # scan can be skipped (single-overlap exclusion lemma, SM Sec. I)
        for e in (nsm if len(inter) > 1 else ()):
            if e in nbg:
                j = len(e)
                sharedU += _logchoose(u, j)
                sharedN += _logchoose(N, j)
                if e in own1 and e in own2:
                    sharedA_N += _logchoose(N, j)
        # delta without the cross edges; each cross edge only lowers it by
        # at most logC(N, n(e)), so if delta_wo - allowance >= 0 the pair
        # provably cannot merge and the (comparatively expensive) cross
        # enumeration is skipped entirely (exact pruning).
        delta_wo = (C0 + U1 + U2 - sharedU
                    - ownSelf_c[s1] - ownSelf_c[s2]
                    - (nestN_c[s1] + nestN_c[s2] - sharedN
                       - ownN_c[s1] - ownN_c[s2] + sharedA_N))
        if _sp:
            delta_wo += _sp_term(s1, s2, u)
        sxcInt = sum(nodeCrossN.get(v, 0.0) for v in inter)
        allow = min(sxc[s1], sxc[s2]) - sxcInt
        if delta_wo - allow >= 0:
            return None

        crossU = crossN = 0.0
        _il = len(inter)
        for e in _cross_terms(s1, s2, n1 - _il, n2 - _il):
            j = len(e)
            crossU += _logchoose(u, j)
            crossN += _logchoose(N, j)
        return delta_wo + crossU - crossN

    def _push(s1, s2):
        d = _delta(s1, s2)
        key = frozenset({s1, s2})
        if d is None:
            # provably never merges in the current state; clear any earlier
            # entry for this key so a matching stale heap tuple cannot pop
            # as current later
            if valid.pop(key, None) is not None:
                s_pairs.get(s1, set()).discard(key)
                s_pairs.get(s2, set()).discard(key)
            return
        valid[key] = d
        s_pairs.setdefault(s1, set()).add(key)
        s_pairs.setdefault(s2, set()).add(key)
        heapq.heappush(heap, (d, s1, s2))

    def _invalidate(s):
        for key in list(s_pairs.get(s, ())):
            valid.pop(key, None)
            for t in key:
                if t != s:
                    s_pairs.get(t, set()).discard(key)
        s_pairs.pop(s, None)

    def _drop_state(s):
        for dd in (sfm, nested, histN, nestN_c, ownN_c, ownSelf_c, degsum, sxc):
            dd.pop(s, None)

    # ── initial heap: only pairs sharing >= 1 node (deterministic order) ─────
    # Seeding is quadratic in the number of surviving simplices, and on
    # dense corpora most overlapping pairs are provably never mergeable.
    # For large survivor counts the SOUND lower bound of Eq. (delta-ss)
    # (sharedU <= min(U1,U2); sharedN - sharedA_N >= 0 dropped;
    # crossU - crossN >= -|cross|_cap * maxdec with |cross| bounded by the
    # smaller difference side's incident-edge count) is evaluated for ALL
    # pairs at once with blocked numpy matrix products; only pairs whose
    # bound is negative get an exact delta and a heap entry, in the same
    # deterministic (key-sorted) order as the direct loop. Pairs pruned
    # here can never merge in the initial state and are re-considered
    # automatically if a participant changes, so the merge sequence is
    # unchanged (exact pruning).
    surv = sorted(sfm, key=_ckey)
    S0 = len(surv)

    def _seed_pair_filtered(a, b):
        """Common candidate filter: skip non-overlapping pairs; apply the
        single-overlap restriction (SM Sec. I) uniformly; otherwise push."""
        small_op, big_op = (a, b) if len(a) <= len(b) else (b, a)
        ov = 0
        for v in small_op:
            if v in big_op:
                ov += 1
                if ov > 1:
                    break
        if ov == 0:
            return
        if ov == 1 and _l1_seed_excluded(a, b):
            return
        _push(a, b)

    # Budget clock starts BEFORE seeding (2026-07-22): on hub-heavy
    # million-edge corpora the pre-loop work (candidate generation and
    # exact-delta seeding) can dominate wall-clock, so it must count
    # against budget_s. Seeding is truncatable: stopping early only
    # removes candidates (restriction-only, output remains a valid
    # decomposition).
    _t_start = time.time()
    _seed_deadline = (_t_start + budget_s) if budget_s is not None else None

    if approx:
        # ── approximate seeding: MinHash-LSH + nestedness parents ─────────
        # (proposal Sec. 7.3.1-7.3.2). Candidates come from LSH band
        # collisions over node-set signatures plus each survivor's direct
        # nested supersets; every candidate still goes through the exact
        # delta, so approximation = missed candidates only. Hub buckets
        # are capped at hub_cap lowest-key members.
        rng_lsh = np.random.RandomState(0)
        node_ids = {v: i for i, v in enumerate(edge_index.keys())}
        P_MOD = (1 << 61) - 1
        a_h = rng_lsh.randint(1, P_MOD, size=lsh_num_perm, dtype=np.int64)
        b_h = rng_lsh.randint(0, P_MOD, size=lsh_num_perm, dtype=np.int64)
        # choose (bands, rows) with collision threshold ~ (1/bands)^(1/rows)
        best_br = (lsh_num_perm, 1)
        best_err = float('inf')
        for r_rows in range(1, lsh_num_perm + 1):
            if lsh_num_perm % r_rows:
                continue
            b_bands = lsh_num_perm // r_rows
            thr = (1.0 / b_bands) ** (1.0 / r_rows)
            if abs(thr - lsh_threshold) < best_err:
                best_err = abs(thr - lsh_threshold)
                best_br = (b_bands, r_rows)
        b_bands, r_rows = best_br
        sigs = np.empty((S0, lsh_num_perm), dtype=np.int64)
        for i, sv in enumerate(surv):
            ids = np.fromiter((node_ids[v] for v in sv), dtype=np.int64,
                              count=len(sv))
            sigs[i] = ((a_h[None, :] * ids[:, None] + b_h[None, :])
                       % P_MOD).min(axis=0)
        seen_pairs = set()
        cand_list = []
        for bnd in range(b_bands):
            buckets = {}
            block = sigs[:, bnd * r_rows:(bnd + 1) * r_rows]
            for i in range(S0):
                buckets.setdefault(block[i].tobytes(), []).append(i)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                if len(members) > hub_cap:
                    members = sorted(members)[:hub_cap]
                for ai in range(len(members)):
                    for bi in range(ai + 1, len(members)):
                        kpr = (members[ai], members[bi])
                        if kpr not in seen_pairs:
                            seen_pairs.add(kpr)
                            cand_list.append(kpr)
        # nestedness parents among survivors (rarest-node containment scan).
        # Hub guard (2026-07-22): if even the RAREST node of a survivor
        # sits in more than 64*hub_cap survivors, scanning its list is
        # quadratic poison on heavy-tailed corpora -- skip it (candidates
        # lost only; LSH collisions can still propose such pairs).
        _nested_scan_cap = 64 * hub_cap
        surv_pos_l = {sv: i for i, sv in enumerate(surv)}
        for i, sv in enumerate(surv):
            v_star = min(sv, key=lambda v: len(simplex_index.get(v, ())))
            lst = simplex_index.get(v_star, ())
            if len(lst) > _nested_scan_cap:
                continue
            for t in lst:
                if t is not sv and sv < t:
                    j = surv_pos_l[t]
                    kpr = (min(i, j), max(i, j))
                    if kpr not in seen_pairs:
                        seen_pairs.add(kpr)
                        cand_list.append(kpr)
        cand_list.sort()
        for _ci, (i, j) in enumerate(cand_list):
            if (_seed_deadline is not None and (_ci & 2047) == 0
                    and time.time() > _seed_deadline):
                if logfile:
                    logfile.write("BUDGET: seeding truncated at %d/%d "
                                  "candidates\n" % (_ci, len(cand_list)))
                break
            _seed_pair_filtered(surv[i], surv[j])
    elif S0 < 300:
        seen_pairs = set()
        for s in surv:
            for other in sorted(_candidates(s) - {s}, key=_ckey):
                key = frozenset({s, other})
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                ov = len(s & other)
                if ov == 0:
                    continue
                if ov == 1 and _l1_seed_excluded(s, other):
                    continue
                _push(s, other)
    else:
        # ── nnz-proportional pair accumulation (2026-07-18, exact) ───────
        # The previous implementation built a DENSE S0 x Nn incidence
        # matrix and blocked S0 x S0 matmuls: O(S0 * Nn) memory and
        # O(S0^2 * Nn) flops regardless of sparsity, which is exactly
        # backwards on large corpora (S0 ~ E right after phase 1). Here
        # the same intersection counts and degree-weighted intersections
        # are accumulated only over pairs that actually share a node:
        # for each node, the C(d, 2) survivor pairs through it are
        # generated as index arrays and grouped with np.unique, so cost
        # and memory are proportional to the number of overlapping pair
        # incidences sum_v C(d_v, 2) -- the true output size -- with no
        # dense N dimension anywhere. Bound expressions, pruning, and the
        # deterministic evaluation order are unchanged.
        n_arr = np.array([len(sv) for sv in surv], dtype=np.int64)
        oS = np.array([ownSelf_c[sv] for sv in surv])
        oN = np.array([ownN_c[sv] for sv in surv])
        nN = np.array([nestN_c[sv] for sv in surv])
        ds = np.array([degsum[sv] for sv in surv])
        sx = np.array([sxc[sv] for sv in surv])
        # size-prior bookkeeping: operand size cost recovered only for
        # non-singleton operands (see _sp_term)
        spn = (n_arr * np.array([len(sfm[sv]) > 1 for sv in surv])
               if _sp else np.zeros_like(n_arr))
        surv_pos = {sv: i for i, sv in enumerate(surv)}
        kk = kmax
        u_max = int(2 * n_arr.max())
        LCu = np.zeros((u_max + 1, kk + 1))
        for u in range(u_max + 1):
            for j in range(min(u, kk) + 1):
                LCu[u, j] = _logchoose(u, j)
        LCNu = np.array([_logchoose(N, u) for u in range(u_max + 1)])
        LCNj = np.array([_logchoose(N, j) for j in range(kk + 1)])
        Hm = np.zeros((S0, kk + 1))
        for i, sv in enumerate(surv):
            for j, c in histN[sv].items():
                Hm[i, j] = c
        Utab = Hm @ LCu.T                       # S0 x (u_max+1)
        jm = np.minimum(np.arange(u_max + 1), kk)
        maxdec = LCNj[jm] - LCu[np.arange(u_max + 1), jm]
        # per-node pair generation: for each node, the survivors through it
        # (as sorted row indices), pairs via triu indices
        chunks_i, chunks_j, chunks_w = [], [], []
        for v, simps_v in simplex_index.items():
            d = len(simps_v)
            if d < 2:
                continue
            m = np.sort(np.fromiter((surv_pos[t] for t in simps_v),
                                    dtype=np.int64, count=d))
            npairs_v = d * (d - 1) // 2
            if node_pair_cap is not None and npairs_v > node_pair_cap:
                # ── per-STRUCTURE stochastic shortcut (2026-07-28) ────────
                # A single hub node of survivor-degree d contributes
                # C(d,2) candidate pairs, so one heavy structure -- not
                # system size -- is what makes exact seeding quadratic.
                # For such nodes only, draw a seeded uniform subsample of
                # ~node_pair_cap of their pairs (deduplicated within the
                # node); all other nodes are enumerated exhaustively, so
                # cost is sum_v min(C(d_v,2), node_pair_cap). SOUND:
                # every surviving pair still passes through the exact
                # delta, so subsampling can only MISS candidate merges.
                # Pairs overlapping in >=2 nodes are regenerated at each
                # shared node and recoverable via post-merge re-pairing;
                # what is thinned is dominated by single-overlap hub
                # pairs, which the l1 seeding restriction screens hardest
                # anyway. Seeded per node (crc32 of the node label), so
                # runs are reproducible independent of hash randomization.
                rng = np.random.default_rng(
                    zlib.crc32(repr(v).encode('utf8', 'replace')))
                a = rng.integers(0, d, size=2 * node_pair_cap)
                b = rng.integers(0, d, size=2 * node_pair_cap)
                lo = np.minimum(a, b); hi = np.maximum(a, b)
                ok = lo < hi
                lo, hi = lo[ok], hi[ok]
                # dedupe within this node: an intra-node duplicate would
                # inflate the pair's shared-node count `it` below
                uniq = np.unique(lo * d + hi)
                ti, tj = uniq // d, uniq % d
            else:
                ti, tj = np.triu_indices(d, 1)
            chunks_i.append(m[ti]); chunks_j.append(m[tj])
            chunks_w.append(np.full(ti.shape, float(deg.get(v, 0))))
        if chunks_i:
            pi = np.concatenate(chunks_i); pj = np.concatenate(chunks_j)
            pw = np.concatenate(chunks_w)
            codes = pi * S0 + pj
            ucodes, inv, it = np.unique(codes, return_inverse=True,
                                        return_counts=True)
            dsInt = np.bincount(inv, weights=pw, minlength=len(ucodes))
            ii = (ucodes // S0).astype(np.int64)
            jj = (ucodes % S0).astype(np.int64)
            u = n_arr[ii] + n_arr[jj] - it
            C0 = LCNu[u] - LCNu[n_arr[ii]] - LCNu[n_arr[jj]]
            U1 = Utab[ii, u]
            U2 = Utab[jj, u]
            cc = np.minimum(ds[ii] - dsInt, ds[jj] - dsInt)
            base = -oS[ii] - oS[jj] - nN[ii] - nN[jj] + oN[ii] + oN[jj]
            if _sp:
                # exact size-prior contribution (same for bound and delta)
                base = base + u - spn[ii] - spn[jj]
            # generic bound: sharedU <= min(U1,U2); cross charge bounded by
            # either the count*maxdec allowance or the per-node exact-size
            # allowance min(sxc1, sxc2) -- both sound, take the tighter
            sxa = np.minimum(sx[ii], sx[jj])
            allow = np.minimum(np.maximum(cc, 0) * maxdec[u], sxa)
            bound = C0 + np.maximum(U1, U2) + base - allow
            # ── single-overlap seeding restriction (SM Sec. I) ────────────
            # When |s1 & s2| == 1 no hyperedge fits inside the intersection,
            # so the shared-face terms vanish IDENTICALLY and the delta
            # decomposes into exactly computable per-simplex terms plus the
            # cross-edge credit. A pair sharing a single node is seeded only
            # if the exactly computable part alone is negative (the cross
            # credit is forfeited at seeding; it remains available to every
            # pair re-evaluated after either participant changes). Removes
            # the hub-node pair explosion on heavy-tailed corpora.
            l1 = it == 1
            if l1.any():
                excl_l1 = C0 + U1 + U2 + base
                bound = np.where(l1, excl_l1, bound)
            keep = bound < 0
            ki, kj, kb = ii[keep], jj[keep], bound[keep]
            # ── per-OBJECT seeding candidate budget (2026-07-29) ──────────
            # The bound-survivor list itself was unbounded: on hub-heavy
            # corpora (amplified by the geometric prior's -ell size bonus,
            # which makes overlap bounds more negative) it reached
            # millions of entries -- gigabytes of candidate storage plus
            # millions of exact delta evaluations, the OOM-killed-worker
            # slowdown of the 2026-07-29 corpus runs. Apply the SAME
            # per-object budget used at re-pairing: each object enters
            # the heap with at most its hub_cap best candidates by the
            # sound lower bound. Deterministic (ties broken by pair
            # index); restriction-only -- a pruned pair re-enters via
            # post-merge re-pairing as soon as either participant
            # changes. Exact-delta work is now <= hub_cap * S0.
            if len(ki) > 0:
                def _group_ranks(g, b):
                    o = np.lexsort((np.arange(len(g)), b, g))
                    gs = g[o]
                    starts = np.flatnonzero(
                        np.concatenate(([True], gs[1:] != gs[:-1])))
                    grp = np.zeros(len(g), dtype=np.int64)
                    grp[starts] = 1
                    grp = np.cumsum(grp) - 1
                    first = starts[grp]
                    r = np.empty(len(g), dtype=np.int64)
                    r[o] = np.arange(len(g)) - first
                    return r
                r_i = _group_ranks(ki, kb)
                r_j = _group_ranks(kj, kb)
                keep2 = (r_i < hub_cap) | (r_j < hub_cap)
                ki, kj = ki[keep2], kj[keep2]
            # exact evaluation in the deterministic order of the direct
            # double loop (ascending first index, then second); arrays,
            # not python tuples, so survivor storage stays compact
            o = np.lexsort((kj, ki))
            ki, kj = ki[o], kj[o]
            for _ci in range(len(ki)):
                if (_seed_deadline is not None and (_ci & 2047) == 0
                        and time.time() > _seed_deadline):
                    if logfile:
                        logfile.write("BUDGET: seeding truncated at %d/%d "
                                      "candidates\n" % (_ci, len(ki)))
                    break
                _push(surv[ki[_ci]], surv[kj[_ci]])

    itr = 0
    n_pops = 0
    _no_l1_repair = False
    _budget_stage = 0
    _rng_sto = random.Random(0)
    while heap:
        d, s1, s2 = heapq.heappop(heap)
        n_pops += 1
        # ── optional two-stage wall-clock budget (proposal Sec. 7.4):
        # stage 1 (at budget_s): tighten the candidate knobs -- tightening
        # only ever RESTRICTS which candidates get pushed, so state stays
        # consistent; stage 2 (at 2*budget_s): stop merging outright and
        # finalize -- every intermediate greedy state is a valid
        # decomposition, so the result is simply the best found in budget.
        if budget_s is not None and n_pops % 1000 == 0 \
                and time.time() - _t_start > budget_s:
            if _budget_stage == 0:
                if theta == 0.0:
                    theta = 1.0
                hub_cap = max(1, hub_cap // 2)
                _no_l1_repair = True
                _budget_stage = 1
                budget_s = 2 * budget_s   # stage-2 deadline (from start)
                if logfile:
                    logfile.write("BUDGET stage 1: theta=1, hub_cap halved, "
                                  "overlap-1 re-pairing disabled\n")
            else:
                _budget_stage = 2
                if logfile:
                    logfile.write("BUDGET stage 2: merge loop stopped; "
                                  "finalizing best-found decomposition\n")
                break
        key = frozenset({s1, s2})
        if key not in valid or valid[key] != d:   # stale heap entry
            continue
        if d >= 0:
            break                                  # no beneficial merge left
        # minimum-improvement threshold (approx mode; SWeG-style decay).
        # theta_t shrinks as merges accumulate, so early merges must clear
        # a higher bar; rejected entries are dropped (restriction only).
        if theta > 0.0 and d >= -(theta / (1.0 + itr)):
            valid.pop(key, None)
            s_pairs.get(s1, set()).discard(key)
            s_pairs.get(s2, set()).discard(key)
            continue
        if s1 not in sfm or s2 not in sfm:
            continue

        union = s1 | s2
        u = len(union)
        if size_cap is not None and u > size_cap:
            # giant-simplex guard (approx mode): skip merges whose union
            # would exceed size_cap; drop the entry so it cannot re-pop
            valid.pop(key, None)
            s_pairs.get(s1, set()).discard(key)
            s_pairs.get(s2, set()).discard(key)
            continue

        # ── build the union's state incrementally (steal the larger set) ──
        _ns1, _ns2 = len(s1), len(s2)
        _il = len(s1 & s2)
        cross = _cross_terms(s1, s2, _ns1 - _il, _ns2 - _il)
        nsm, nbg = nested[s1], nested[s2]
        s_small, s_big = (s1, s2) if len(nsm) <= len(nbg) else (s2, s1)
        if len(nsm) > len(nbg):
            nsm, nbg = nbg, nsm
        hist_u = histN[s_big]
        shared_sizes = Counter()
        for e in nsm:
            if e in nbg:
                shared_sizes[len(e)] += 1
        nested_u = nbg                       # steal
        nested_u.update(nsm)
        nested_u.update(cross)
        hist_u = hist_u + histN[s_small] - shared_sizes
        for e in cross:
            hist_u[len(e)] += 1
        crossN_sum = sum(_logchoose(N, len(e)) for e in cross)
        sharedN_sum = sum(c * _logchoose(N, j) for j, c in shared_sizes.items())
        nestN_u = nestN_c[s1] + nestN_c[s2] - sharedN_sum + crossN_sum
        inter_nodes = [v for v in (s1 if len(s1) <= len(s2) else s2)
                       if v in (s2 if len(s1) <= len(s2) else s1)]
        degsum_u = degsum[s1] + degsum[s2] - sum(deg.get(v, 0) for v in inter_nodes)
        sxc_u = sxc[s1] + sxc[s2] - sum(nodeCrossN.get(v, 0.0) for v in inter_nodes)

        # Remove merged simplices from all data structures. Light-node
        # neighbors are carried in nbrs (steal the larger, drop dead
        # entries); heavy-node neighbors are resolved on demand from the
        # incrementally maintained simplex_index below.
        _invalidate(s1); _invalidate(s2)
        light_u = nbrs.pop(s1, set()) if s1 != union else set(nbrs.get(s1, ()))
        light_2 = nbrs.pop(s2, set()) if s2 != union else set(nbrs.get(s2, ()))
        if len(light_u) < len(light_2):
            light_u, light_2 = light_2, light_u
        light_u.update(light_2)
        light_u.discard(s1); light_u.discard(s2); light_u.discard(union)
        if s1 != union:
            del sfm[s1]
        if s2 != union:
            del sfm[s2]
        for dd in (nested, histN, nestN_c, ownN_c, ownSelf_c, degsum, sxc):
            if s1 != union:
                dd.pop(s1, None)
            if s2 != union:
                dd.pop(s2, None)

        # incremental simplex_index maintenance: retire the operands...
        for op in (s1, s2):
            if op != union:
                for v in op:
                    simplex_index[v].discard(op)

        # Register the union (owned == nested for merge-born simplices,
        # matching the previous sfm[union] = all-nested behavior)
        sfm[union] = nested_u
        nested[union] = nested_u
        histN[union] = hist_u
        nestN_c[union] = nestN_u
        ownN_c[union] = nestN_u
        ownSelf_c[union] = sum(c * _logchoose(u, j) for j, c in hist_u.items())
        degsum[union] = degsum_u
        sxc[union] = sxc_u
        s_pairs.setdefault(union, set())
        hn_u = hnodes.pop(s1, frozenset()) | hnodes.pop(s2, frozenset())
        hnodes[union] = hn_u
        # ...and index the union
        for v in union:
            simplex_index[v].add(union)

        # heavy-node neighbors, on demand from the live index (few heavy
        # nodes; entries are current thanks to the incremental maintenance)
        heavy_live = set()
        for v in hn_u:
            heavy_live |= simplex_index.get(v, set())
        heavy_live.discard(union)

        cand_all = light_u | heavy_live

        # Remove simplices now strictly nested inside the union (any such k
        # shares every node with the union, hence appears among the light
        # neighbors or a heavy node's index entry)
        nested_simps = [k for k in cand_all if k in sfm and k < union]
        for k in nested_simps:
            _invalidate(k)
            del sfm[k]
            _drop_state(k)
            nbrs.pop(k, None)
            hnodes.pop(k, None)
            for v in k:
                simplex_index[v].discard(k)
            cand_all.discard(k)
            light_u.discard(k)

        itr += 1
        if logfile:
            logfile.write(f"Phase2 iter {itr}: |{len(s1)}|+|{len(s2)}|->|{u}|  delta={d:.3f}\n")

        # Register the union's light neighbors (compacted to live entries)
        # and mirror the union into the light neighbors' own sets; heavy
        # neighbors intentionally NOT stored -- future lookups will find
        # the union through the heavy nodes' index entries.
        light_live = [x for x in light_u if x in sfm]
        nbrs[union] = set(light_live)
        for x in light_live:
            nbrs.setdefault(x, set()).add(union)

        # Post-merge re-pairing, with the SAME single-overlap restriction
        # as seeding (SM Sec. I, applied uniformly): a hub-touching union
        # would otherwise re-pair with every simplex at the hub each time
        # it grows. Overlap size is computed over the smaller operand.
        live = [x for x in cand_all if x in sfm]
        live_sorted = sorted(live, key=_ckey)
        # per-union re-pairing budget (2026-07-29 v13): BOTH candidate
        # classes are budgeted at hub_cap. The previous version kept all
        # overlap->=2 neighbors on the assumption they are few -- false
        # on hub-dense networks (ingredients, cast members, tags), where
        # a growing union overlaps hundreds of objects in >=2 nodes
        # after EVERY merge; profiling showed 2e5 exact deltas (~90% of
        # runtime, unbounded heap/pair bookkeeping = the memory crashes)
        # on a 1,200-edge cooking network. Each class independently
        # keeps a seeded random sample of hub_cap when larger --
        # restriction-only, and re-pairing recurs at every subsequent
        # merge, so coverage accumulates across the run. (stochastic_eps
        # is retained in the signature for backward compatibility but is
        # no longer used.)
        _l1_c, _ge2_c = [], []
        for s_other in live_sorted:
            small_op = s_other if len(s_other) <= u else union
            big_op = union if small_op is s_other else s_other
            ov = 0
            for v in small_op:
                if v in big_op:
                    ov += 1
                    if ov > 1:
                        break
            if ov == 0:
                continue
            (_l1_c if ov == 1 else _ge2_c).append(s_other)
        if _no_l1_repair:
            _l1_c = []
        if len(_ge2_c) > hub_cap:
            _ge2_c = _rng_sto.sample(_ge2_c, hub_cap)
        if len(_l1_c) > hub_cap:
            _l1_c = _rng_sto.sample(_l1_c, hub_cap)
        for s_other in _ge2_c:
            _push(union, s_other)
        for s_other in _l1_c:
            if _l1_seed_excluded(union, s_other):
                # clear any stale entry for this key (same reasoning as in
                # _push's pruning branch)
                key = frozenset({union, s_other})
                if valid.pop(key, None) is not None:
                    s_pairs.get(union, set()).discard(key)
                    s_pairs.get(s_other, set()).discard(key)
                continue
            _push(union, s_other)

        # ── heap hygiene: purge stale tuples when they dominate ─────────
        if len(heap) > max(4 * len(valid), 1024):
            heap = [(dv, k1, k2) for k, dv in valid.items()
                    for k1, k2 in [tuple(sorted(k, key=_ckey))]]
            heapq.heapify(heap)

    # record budget outcome for callers (last_run_info is (re)initialized by
    # greedy_optimizer before phase2 runs; harmless if phase2 is called
    # directly in tests)
    last_run_info['budget_stage'] = _budget_stage
    last_run_info['n_pops'] = n_pops
    last_run_info['n_merges'] = itr
    return {s: frozenset(faces) for s, faces in sfm.items()}


def greedy_optimizer(G_list, logpath='/dev/null', N=None, mode='auto',
                     pair_cap=5_000_000, edge_cap=2_000_000,
                     hub_cap=None, size_cap=None,
                     lsh_num_perm=None, lsh_threshold=None,
                     stochastic_eps=0.0, theta=0.0, budget_s=None,
                     node_pair_cap='auto', seed_mem_cap=None):
    """Two-phase greedy MDL optimizer. Returns (S, H, DL, eta).

    By default N is inferred as the number of distinct nodes participating
    in at least one hyperedge of G_list. Pass an explicit N to override
    this -- e.g. when G_list is a small hyperedge set embedded in a much
    larger ambient network (only the node-set prior costs log C(N, n(s))
    depend on N; neither phase enumerates nodes, so this is a pure
    parameter override with no change to existing call sites, all of
    which omit N and get the original inferred behavior).

    mode:
      'exact'  -- exhaustive candidate enumeration (previous behavior).
      'approx' -- MinHash-LSH candidate seeding with per-hub caps and
                  optional merge thresholds (see phase2); every proposed
                  merge is still evaluated with the exact delta, so the
                  approximation can only MISS merges, never execute an
                  unsound one.
      'auto'   -- measure complexity after phase 1 and switch to 'approx'
                  when the overlapping-pair upper bound sum_v C(d_v, 2)
                  exceeds pair_cap, E exceeds edge_cap, or the maximum
                  node simplex-degree exceeds 10*hub_cap.

    node_pair_cap (2026-07-28, per-structure stochastic shortcut): in
    exact seeding, any single node whose survivor-degree d gives
    C(d,2) > node_pair_cap contributes only a seeded uniform subsample of
    ~node_pair_cap of its candidate pairs; all other nodes are enumerated
    exhaustively. The shortcut therefore engages per heavy STRUCTURE, not
    per system: small systems (max d ~ sqrt(2*cap), ~45 at the default)
    are bit-identical to exhaustive enumeration, and seeding cost is
    bounded by sum_v min(C(d_v,2), cap) -- smooth in system size, with no
    global mode switch. Since every surviving candidate is still scored
    with the exact merge delta, subsampling can only miss merges, never
    execute an unsound one. With the cap active, 'auto' escalates to the
    LSH 'approx' path on a single MEMORY criterion: the exact seeding
    tables' estimated footprint (pair arrays + dense Hm/Utab, computed
    from the post-phase-1 degree histogram) exceeding seed_mem_cap
    bytes. No wall-clock is involved, and escalated instances run the
    default LSH settings (no per-tier presets). Those defaults follow
    two stated rules rather than tuning: collision threshold 1/2 (the
    parameter-free majority-similarity rule -- propose a pair iff its
    node sets are more similar than dissimilar) and signature length
    k = 25 = 1/(4*delta^2), fixing the MinHash similarity estimator's
    standard error at the threshold to delta = 0.1. Lean by design: the
    escalation path exists FOR memory (128-perm signatures cost 5x more;
    2026-07-29 crash postmortem). The method thus carries
    exactly three resource constants: node_pair_cap (per-node seeding
    pair budget), hub_cap (per-union re-pairing budget, also the LSH
    bucket cap), and seed_mem_cap (bytes). The legacy pair_est/dmax/
    edge_cap triggers apply only when node_pair_cap=None, which also
    restores the previous behavior exactly.

    The mode actually used and its triggering metrics are recorded in the
    module-level dict `last_run_info`.

    Arguments left at their sentinels resolve to the module-level
    resource constants (env-overridable; see the block at the top of
    this file): hub_cap/lsh_num_perm/lsh_threshold/seed_mem_cap = None,
    node_pair_cap = 'auto' ('auto' -> NODE_PAIR_CAP; an explicit None
    still selects the legacy pre-cap behavior).
    """
    if hub_cap is None:
        hub_cap = HUB_CAP
    if lsh_num_perm is None:
        lsh_num_perm = LSH_NUM_PERM
    if lsh_threshold is None:
        lsh_threshold = LSH_THRESHOLD
    if seed_mem_cap is None:
        seed_mem_cap = SEED_MEM_CAP
    if node_pair_cap == 'auto':
        node_pair_cap = NODE_PAIR_CAP
    G_frozen = frozenset(frozenset(e) for e in G_list)
    if N is None:
        N = len(set(node for e in G_frozen for node in e))
    verbose_log = os.path.basename(logpath) != os.path.basename(os.devnull)
    with open(logpath, "w") as lf:
        lf.write(f"N={N}  E={len(G_frozen)}  L0={baseline_DL(G_frozen, N):.3f}\n\n")
        lf.write("=== Phase 1 ===\n")
        sfm = phase1(G_frozen, N, logfile=lf if verbose_log else None)
        # ── complexity measure and auto switch (proposal Sec. 7.2) ───────
        deg_S = Counter(v for s_ in sfm for v in s_)
        pair_est = sum(dd * (dd - 1) // 2 for dd in deg_S.values())
        E_ = len(G_frozen)
        dmax = max(deg_S.values(), default=0)
        if node_pair_cap is not None:
            # per-node subsampling bounds seeding COST structurally, but
            # the exact vectorized seeding still materializes its tables:
            # the pair arrays (~24 B per generated pair entry) and the
            # dense per-simplex tables Hm (S0 x (kmax+1)) and Utab
            # (S0 x (2*n_max+1)), 8 B each. 2026-07-29: estimate those
            # bytes up front and escalate to the LSH generator on MEMORY
            # grounds when they would exceed seed_mem_cap -- the same
            # justification as the E > edge_cap trigger, per instance
            # rather than at an arbitrary size threshold. (Concurrent
            # multi-GB seeding workers crashed the 2026-07-28 overnight
            # corpus run.) No wall-clock enters anywhere.
            S0_ = len(sfm)
            nmax_ = max((len(s_) for s_ in sfm), default=0)
            kmax_ = max((len(e) for e in G_frozen), default=0)
            pairs_capped = sum(min(dd * (dd - 1) // 2, node_pair_cap)
                               for dd in deg_S.values())
            seed_mem_est = (24 * pairs_capped
                            + 8 * S0_ * (kmax_ + 1)
                            + 8 * S0_ * (2 * nmax_ + 1)) if S0_ >= 300 else 0
            # single escalation criterion (2026-07-29 consolidation):
            # the estimated memory footprint alone decides -- any
            # million-edge instance blows the estimate anyway, so the
            # old E > edge_cap trigger was redundant with it
            approx = (mode == 'approx') or (mode == 'auto' and
                      seed_mem_est > seed_mem_cap)
            trigger = ('forced' if mode == 'approx' else
                       'seed_mem' if seed_mem_est > seed_mem_cap
                       else None) if approx else None
        else:
            seed_mem_est = None
            approx = (mode == 'approx') or (mode == 'auto' and
                      (pair_est > pair_cap or E_ > edge_cap or dmax > 10 * hub_cap))
            trigger = ('forced' if mode == 'approx' else
                       'pair_est' if pair_est > pair_cap else
                       'edge_cap' if E_ > edge_cap else
                       'hub_deg' if dmax > 10 * hub_cap else None) if approx else None
        last_run_info.clear()
        last_run_info.update(mode='approx' if approx else 'exact',
                             requested_mode=mode, pair_est=pair_est, E=E_,
                             S0=len(sfm), dmax=dmax, trigger=trigger,
                             hub_cap=hub_cap, size_cap=size_cap,
                             node_pair_cap=node_pair_cap,
                             seed_mem_est=seed_mem_est,
                             lsh_num_perm=lsh_num_perm,
                             lsh_threshold=lsh_threshold, theta=theta)
        lf.write(f"mode={last_run_info['mode']} (requested {mode}; "
                 f"pair_est={pair_est} E={E_} dmax={dmax} trigger={trigger})\n")
        # description_length(..., list(sfm), N) here is diagnostic only (it
        # isn't used for anything but this log line) and costs O(E * |S|)
        # even with the indexed _assign_faces helper once |S| is large --
        # right after phase 1 is exactly when |S| is largest, before phase 2
        # has had a chance to shrink it. Skip it unless a real log file was
        # requested (logpath defaults to /dev/null on every empirical-corpus
        # call, so this was previously paid on every single network).
        if verbose_log:
            lf.write(f"After Phase 1: |S|={len(sfm)}  DL={description_length(G_frozen, list(sfm), N):.3f}\n\n")
        else:
            lf.write(f"After Phase 1: |S|={len(sfm)}\n\n")
        lf.write("=== Phase 2 ===\n")
        sfm = phase2(sfm, G_frozen, N, logfile=lf if verbose_log else None,
                     approx=approx, hub_cap=hub_cap, size_cap=size_cap,
                     lsh_num_perm=lsh_num_perm, lsh_threshold=lsh_threshold,
                     stochastic_eps=stochastic_eps, theta=theta,
                     budget_s=budget_s, node_pair_cap=node_pair_cap)
        L0 = baseline_DL(G_frozen, N)
        DL = description_length(G_frozen, list(sfm), N)
        if DL >= L0:
            sfm = {}; DL = L0
        lf.write(f"Final: |S|={len(sfm)}  DL={DL:.3f}\n")
    return _finalise(G_frozen, list(sfm), DL, L0)


# ── convenience wrapper ──────────────────────────────────────────────────────

def decompose(edges, N=None):
    """Decompose a hypergraph into latent simplices and residual edges.

    Parameters
    ----------
    edges : iterable of node collections (any hashable node labels).
        Hyperedges of size < 2 and duplicates are dropped.
    N : int, optional
        Number of nodes in the ambient system (default: number of
        distinct nodes appearing in `edges`).

    Returns
    -------
    dict with keys 'S' (list of simplex node sets), 'H' (list of
    residual hyperedges), 'F' (dict simplex -> hosted hyperedges),
    'L' (description length, bits), 'L0' (null description length),
    and 'eta' (compression ratio L / L0).
    """
    G_list = [frozenset(e) for e in edges]
    G_list = sorted(set(e for e in G_list if len(e) >= 2),
                    key=lambda e: (len(e), tuple(sorted(e))))
    S, H, L, eta = greedy_optimizer(G_list, N=N, logpath=os.devnull)
    G_frozen = frozenset(G_list)
    if N is None:
        N = len(set(v for e in G_frozen for v in e))
    faces, _ = _assign_faces(G_frozen, S)
    return {'S': [frozenset(s) for s in S],
            'H': sorted(H, key=lambda h: (-len(h), tuple(sorted(h)))),
            'F': {frozenset(s): sorted(faces[s], key=lambda e: (-len(e), tuple(sorted(e))))
                  for s in S},
            'L': L, 'L0': baseline_DL(G_frozen, N), 'eta': eta}
