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
F(s) is the set of observed hyperedges hosted by simplex s. The decomposition is
found with the two-phase greedy algorithm of the paper, consisting of a bottom-up
collapse of the nestedness DAG (Phase 1) followed by greedy merging
of overlapping objects (Phase 2).

This code includes the three resource budgets for scalability to very large networks (a
per-node candidate-pair budget, a per-object candidate budget, and a
memory bound under which candidate seeding switches to a MinHash-LSH
generator).

Quick start:

    from simplicial_decomposition import decompose
    result = decompose(hyperedges)          # iterable of node collections
    result['S']      # inferred simplices (list of frozensets)
    result['H']      # residual hyperedges (list of frozensets)
    result['eta']    # compression ratio L / L0 (1.0 = no simplicial structure)

Dependencies: numpy. (Python >= 3.9.)
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
    """log2 C(n, k)"""
    if k < 0 or k > n:
        return float('inf')
    return logchoose(n, k) / math.log(2)


def _enclosing_key(s):
    """Canonical tie-break key for selecting the smallest enclosing simplex:
    size first, then lexicographic on the sorted node tuple. Guarantees a
    deterministic choice whenever multiple candidates in S have equal size."""
    return (len(s), tuple(sorted(s)))


def _assign_faces(G_frozen, S):
    """Assign each edge in G_frozen to its smallest enclosing simplex in S
    (ties broken via _enclosing_key)
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
    """Computes full description length objective L(S, H)."""
    S = list(S)
    faces_assigned, covered = _assign_faces(G_frozen, S)
    H = G_frozen - covered
    dl = (sum(_logchoose(N, len(h)) for h in H)
          + sum(_logchoose(N, len(s)) + sum(_logchoose(len(s), len(e)) for e in faces_assigned[s])
                for s in S))
    if _size_prior_on():
        # geometric size prior: n(s) bits per non-singleton simplex.
        dl += sum(len(s) for s in S if set(faces_assigned[s]) != {s})
    return dl


def baseline_DL(G_frozen, N):
    """L0: description length when every edge is isolated (no simplices,
    S = empty set)"""
    return sum(_logchoose(N, len(e)) for e in G_frozen)


def _finalise(G_frozen, S, DL, L0):
    """Drop neutral singletons (A(s) = {s}), then return
    (S, H, DL, eta)."""
    faces_assigned, _ = _assign_faces(G_frozen, S)
    S = [s for s in S if set(faces_assigned[s]) != {s}]
    _, covered = _assign_faces(G_frozen, S)
    H = G_frozen - covered
    return S, H, DL, DL / L0 if L0 > 0 else float('nan')


def _build_nestedness_dag(G_list):
    """Return child -> set-of-direct-parents in the nestedness partial order
    (h1 -> h2 when h1 is a proper subset of h2, with no intermediate
    hyperedge in G).
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
    direct = {}
    for child, pars in parents.items():
        kept = []
        for p in sorted(pars, key=_enclosing_key):
            if not any(q < p for q in kept):
                kept.append(p)
        direct[child] = set(kept)
    return direct


def phase1(G_frozen, N, logfile=None):
    """Bottom-up DAG collapse, absorbing each edge into
    whichever direct parent yields the most-negative re-encoding gain.
    Parents are iterated in a fixed deterministic order (size, then sorted
    node tuple), and all viable parents are evaluated before committing to
    the one with the smallest delta.
    """
    G_list = list(G_frozen)
    sfm = {h: frozenset({h}) for h in G_list}
    dag = _build_nestedness_dag(G_list)
    # Per-simplex face-SIZE histograms
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
    """Intersection-graph greedy merges via min-heap ("Phase 2")."""
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
    # per-node allowance for the cross-edge charge
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

    # node -> live simplices index; maintained incrementally through merges
    simplex_index = defaultdict(set)
    for s in simplex_faces_map:
        for v in s:
            simplex_index[v].add(s)

    for s, faces in simplex_faces_map.items():
        sfm[s] = set(faces)
        nested[s] = set()
        s_pairs[s] = set()

    # ── nested-set initialization ───────────────────────────────────────────
    # For each edge, scan only the simplex list of its rarest node: any
    # simplex containing the edge appears there.
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

    # ── phase-2 entry invariant ─────────────────────────────────────────────
    # The merge deltas below charge every nested face not owned by an
    # operand at its isolated cost, which is exact only when the owner is
    # a singleton. Absorb every real simplex nested inside another live
    # object into its smallest enclosing object.
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

    # ── collective absorption pass ──────────────────────────────────────────
    # A singleton parent's first absorption pays its full size cost, so
    # phase 1 declines children one at a time even when absorbing them
    # jointly is favorable. For each inclusion-maximal singleton parent,
    # absorb all nested singleton children iff the joint delta is negative.
    _coll = 0
    _sp_on = _size_prior_on()
    for p in sorted(list(sfm), key=lambda t: (len(t), _enclosing_key(t))):
        if p not in sfm or len(sfm[p]) != 1:
            continue
        # inclusion-maximal live parents only: children of non-maximal
        # parents are collected by their enclosing parent later in the sweep
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

    # Overlap-neighbor sets, maintained incrementally:
    # nbrs[union] = nbrs[s1] | nbrs[s2], filtered for liveness at use.
    # Only neighbors through LIGHT nodes (simplex-degree <= PHASE2_HUB_TAU)
    # are stored; neighbors through heavy nodes are resolved on demand
    # from simplex_index.
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

    # canonical sort key per simplex, cached
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
        """Single-overlap candidate restriction, applied
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
        # |s1 & s2| <= 1: no hyperedge fits inside the intersection, so the
        # shared-face terms vanish and the scan is skipped
        for e in (nsm if len(inter) > 1 else ()):
            if e in nbg:
                j = len(e)
                sharedU += _logchoose(u, j)
                sharedN += _logchoose(N, j)
                if e in own1 and e in own2:
                    sharedA_N += _logchoose(N, j)
        # each cross edge lowers delta by at most logC(N, n(e)): if
        # delta_wo - allowance >= 0 the pair provably cannot merge
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
            # provably never merges in the current state; clear any stale
            # entry for this key
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

    # ── initial heap ────────────────────────────────────────────────────────
    # A sound vectorized lower bound on the merge delta is evaluated for
    # all overlapping pairs; only pairs whose bound is negative get an
    # exact delta and a heap entry (exact pruning, deterministic order).
    surv = sorted(sfm, key=_ckey)
    S0 = len(surv)

    def _seed_pair_filtered(a, b):
        """Common candidate filter: skip non-overlapping pairs; apply the
        single-overlap restriction uniformly; otherwise push."""
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

    # the optional wall-clock budget counts seeding as well; stopping
    # early only removes candidates
    _t_start = time.time()
    _seed_deadline = (_t_start + budget_s) if budget_s is not None else None

    if approx:
        # ── approximate seeding: MinHash-LSH + nestedness parents ───────────────
        # Candidates come from LSH band collisions over node-set signatures
        # plus each survivor's direct nested supersets; every candidate is
        # still scored with the exact delta. Hub buckets capped at hub_cap.
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
        # nestedness parents among survivors (rarest-node containment scan);
        # skip nodes whose survivor list is very large (candidates lost only)
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
        # ── pair accumulation over overlapping pairs only ───────────────────────
        # For each node, the survivor pairs through it are generated as index
        # arrays and grouped with np.unique, so cost is proportional to
        # sum_v C(d_v, 2).
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
                # ── per-node candidate-pair budget ──────────────────────────────────────
                # A hub node of survivor-degree d contributes C(d,2) pairs;
                # nodes over node_pair_cap contribute a seeded uniform
                # subsample instead. Subsampling can only miss candidate
                # merges, never execute unsound ones.
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
            # sound generic bound: sharedU <= min(U1, U2); cross charge bounded
            # by the allowance terms
            sxa = np.minimum(sx[ii], sx[jj])
            allow = np.minimum(np.maximum(cc, 0) * maxdec[u], sxa)
            bound = C0 + np.maximum(U1, U2) + base - allow
            # ── single-overlap seeding restriction ──────────────────────────────────
            # For |s1 & s2| == 1 the delta decomposes into exactly computable
            # per-simplex terms plus the cross-edge credit; seed only if the
            # exact part alone is negative (the credit remains available at
            # every re-evaluation).
            l1 = it == 1
            if l1.any():
                excl_l1 = C0 + U1 + U2 + base
                bound = np.where(l1, excl_l1, bound)
            keep = bound < 0
            ki, kj, kb = ii[keep], jj[keep], bound[keep]
            # ── per-object candidate budget ─────────────────────────────────────────
            # Each object enters the heap with at most hub_cap candidates,
            # ranked by the sound lower bound; a pruned pair re-enters via
            # post-merge re-pairing as soon as either participant changes.
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
            # double loop
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
        # ── optional two-stage wall-clock budget ────────────────────────────────
        # stage 1: tighten the candidate knobs (restriction-only);
        # stage 2: stop merging and finalize the best decomposition found.
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
        # minimum-improvement threshold (approx mode, decaying); rejected
        # entries are dropped (restriction only)
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
            # giant-simplex guard (approx mode): skip merges whose union would
            # exceed size_cap
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

        # remove merged simplices from all data structures
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

        # register the union (owned == nested for merge-born simplices)
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

        # heavy-node neighbors resolved on demand from the live index
        heavy_live = set()
        for v in hn_u:
            heavy_live |= simplex_index.get(v, set())
        heavy_live.discard(union)

        cand_all = light_u | heavy_live

        # remove simplices now strictly nested inside the union
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

        # register the union's light neighbors and mirror it into theirs
        light_live = [x for x in light_u if x in sfm]
        nbrs[union] = set(light_live)
        for x in light_live:
            nbrs.setdefault(x, set()).add(union)

        # post-merge re-pairing, with the same single-overlap restriction
        # as seeding
        live = [x for x in cand_all if x in sfm]
        live_sorted = sorted(live, key=_ckey)
        # per-union re-pairing budget: both candidate classes (overlap >= 2
        # and single-overlap) are subsampled to hub_cap when larger;
        # re-pairing recurs at every merge
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
                # clear any stale entry for this key
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

    # record budget outcome for callers
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

    N defaults to the number of distinct nodes in G_list; pass an
    explicit N to embed G_list in a larger ambient system.
    mode: 'exact' (exhaustive candidate seeding), 'approx' (MinHash-LSH
    seeding with per-hub caps), or 'auto' (use 'approx' only when the
    estimated memory footprint of exact seeding exceeds seed_mem_cap).
    Resource arguments left at None resolve to the module-level
    constants; budget_s optionally imposes a two-stage wall-clock
    budget.
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
        # ── complexity estimate and auto mode switch ────────────────────────────
        deg_S = Counter(v for s_ in sfm for v in s_)
        pair_est = sum(dd * (dd - 1) // 2 for dd in deg_S.values())
        E_ = len(G_frozen)
        dmax = max(deg_S.values(), default=0)
        if node_pair_cap is not None:
            # exact seeding materializes its pair arrays; estimate their memory
            # footprint from the survivor degree histogram
            S0_ = len(sfm)
            nmax_ = max((len(s_) for s_ in sfm), default=0)
            kmax_ = max((len(e) for e in G_frozen), default=0)
            pairs_capped = sum(min(dd * (dd - 1) // 2, node_pair_cap)
                               for dd in deg_S.values())
            seed_mem_est = (24 * pairs_capped
                            + 8 * S0_ * (kmax_ + 1)
                            + 8 * S0_ * (2 * nmax_ + 1)) if S0_ >= 300 else 0
            # escalation criterion: the estimated memory footprint alone decides
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
        # diagnostic log only; skipped unless a real log file was requested
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
