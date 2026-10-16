"""
simplicial_decomposition.py -- MDL simplicial decomposition of hypergraphs.

Decomposes an observed hypergraph G into a set S of latent simplices and
a residual set H of independent hyperedges by minimizing the description
length (equivalently, maximizing the Bayesian posterior of the paper's
generative model). Every collection is encoded as a SET, size class by
size class:

    L(S,H) = sum_j log2 C(C(N,j), |H_j|)                      (residual hyperedges)
           + sum_j log2 C(C(N,j), |S_j|)                      (simplex node sets)
           + sum_s [ n(s) + sum_j log2 C(C(n(s),j), |F_j(s)|) ]   (hosted faces)

    L0     = sum_j log2 C(C(N,j), |G_j|)                      (null: uniform random hypergraph)

Here X_j counts objects of size j, n(.) counts nodes, N is the number of nodes,
and F(s) is the set of observed hyperedges hosted by simplex s. Because the
null is a uniform random hypergraph with G's size sequence, random
hypergraphs give eta = L/L0 = 1.

The optimizer is an exact two-phase greedy: collective absorption into
enclosing hyperedges, then best-first merging of overlapping objects, each
followed by a nucleation pass. Every move is scored exactly. Two resource
budgets bound the work: a per-node candidate-pair budget and a per-object
re-pairing budget.

Quick start:

    from simplicial_decomposition import decompose
    result = decompose(hyperedges)          # iterable of node collections
    result['S']      # inferred simplices (list of frozensets)
    result['H']      # residual hyperedges (list of frozensets)
    result['eta']    # compression ratio L / L0 (1.0 = no simplicial structure)

Dependencies: Python >= 3.9 standard library only.
"""

import functools
import heapq
import math
import os
import random
import time
import zlib
from collections import Counter, defaultdict

NODE_PAIR_CAP = int(float(os.environ.get('SD_NODE_PAIR_CAP', 1_000)))
#   per-node seeding pair budget (seeded uniform subsample above it)
HUB_CAP = int(float(os.environ.get('SD_HUB_CAP', 100)))
#   per-object re-pairing budget after each merge

last_run_info = {}


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


def _finalise(G_frozen, S, DL, L0):
    """Drop neutral singletons (A(s) = {s}), then return
    (S, H, DL, eta)."""
    faces_assigned, _ = _assign_faces(G_frozen, S)
    S = [s for s in S if set(faces_assigned[s]) != {s}]
    _, covered = _assign_faces(G_frozen, S)
    H = G_frozen - covered
    return S, H, DL, DL / L0 if L0 > 0 else float('nan')


_LN2 = math.log(2)


@functools.lru_cache(maxsize=1 << 20)
def _lc(n, k):
    """log2 C(n, k) (float n allowed only via _lcc)."""
    if k < 0 or k > n:
        return float('inf')
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / _LN2


def _lfact2(k):
    return math.lgamma(k + 1) / _LN2


@functools.lru_cache(maxsize=1 << 20)
def _lcc(n, j, k):
    """log2 C(C(n, j), k): bits to name a k-subset of the j-subsets of n nodes."""
    if k == 0:
        return 0.0
    lx = _lc(n, j)
    if lx < 40:                                   # X = C(n,j) < ~1e12: exact
        X = math.comb(n, j)
        if k > X:
            return float('inf')
        if k <= 256:
            return (sum(math.log(X - i) for i in range(k))
                    - math.lgamma(k + 1)) / _LN2
        return (math.lgamma(X + 1) - math.lgamma(k + 1)
                - math.lgamma(X - k + 1)) / _LN2
    # X >= 1e12: log (X)_k = k log X + sum_i log(1 - i/X), two-term expansion
    if lx > 900:                                  # corrections < 1e-200 bits
        return k * lx - _lfact2(k)
    X = 2.0 ** lx
    s1 = k * (k - 1) / 2.0
    s2 = (k - 1) * k * (2 * k - 1) / 6.0
    return k * lx - (s1 / X + s2 / (2 * X * X)) / _LN2 - _lfact2(k)


def set_local_cost(n, hist, nest=None):
    """Per-simplex cost: n bits + set code of its faces, size class by size class.

    hist: sizes of the faces this simplex hosts (smallest-enclosing rule).
    nest: sizes of ALL observed hyperedges nested in the simplex. Faces in an
    overlap with a smaller simplex are hosted there, but G is the union of
    what the simplices emit, so this simplex may also emit them. Since
    log C(X, k) is concave in k, emitting all of them or none is optimal in
    each size class, so the class costs min(code(k_own), code(k_nested))."""
    if nest is None:
        return n + sum(_lcc(n, j, c) for j, c in hist.items() if c)
    L = n
    for j, k in nest.items():
        if not k:
            continue
        c = hist.get(j, 0)
        a = _lcc(n, j, c)
        if k > c:
            b = _lcc(n, j, k)
            if b < a:
                a = b
        L += a
    return L


def set_global_cost(N, Hc, Sc):
    return (sum(_lcc(N, j, c) for j, c in Hc.items() if c)
            + sum(_lcc(N, j, c) for j, c in Sc.items() if c))


def set_baseline_DL(G_frozen, N):
    return set_global_cost(N, Counter(len(e) for e in G_frozen), Counter())


def set_description_length(G_frozen, S, N):
    """L(S, H) under the set code, with the manuscript's assignment rule
    (each edge is hosted by its smallest enclosing simplex; ties are broken
    by _enclosing_key; a simplex whose only face is itself is an isolated
    edge). Enclosing simplices may also emit overlap faces (set_local_cost)."""
    S = list(S)
    faces_assigned, covered = _assign_faces(G_frozen, S)
    Hc = Counter(len(e) for e in G_frozen if e not in covered)
    Sc = Counter()
    real = []
    for s in S:
        fa = faces_assigned.get(s, [])
        if not fa:
            continue                      # a simplex with no faces is not part of the model
        if len(fa) == 1 and fa[0] == s:
            Hc[len(s)] += 1               # singleton = isolated edge
            continue
        real.append(s)
    nest = {s: Counter() for s in real}
    node_index = defaultdict(set)
    for s in real:
        for v in s:
            node_index[v].add(s)
    for e in covered:
        it = iter(e)
        cand = set(node_index.get(next(it), ()))
        for v in it:
            cand &= node_index.get(v, set())
        for s in cand:
            if e <= s:
                nest[s][len(e)] += 1
    L = 0.0
    for s in real:
        Sc[len(s)] += 1
        L += set_local_cost(len(s), Counter(len(e) for e in faces_assigned[s]), nest[s])
    return L + set_global_cost(N, Hc, Sc)


class _SetState:
    """Exact face-map state for the set-code objective."""

    def __init__(self, G_frozen, N):
        self.N = N
        self.G = G_frozen
        self.faces = {e: {e} for e in G_frozen}          # object -> owned edges
        self.hist = {e: Counter({len(e): 1}) for e in G_frozen}
        self.simp = {}                                  # simplex -> cached local cost
        # edge -> shared mutable box holding its owner's current key; a merge
        # re-labels the largest operand's box in O(1) rather than
        # re-pointing all of its faces (union-find style)
        self.box = {e: [e] for e in G_frozen}           # object -> its box
        self.obox = {e: self.box[e] for e in G_frozen}  # edge -> owner's box
        self.idx = defaultdict(set)                     # node -> live objects
        for e in G_frozen:
            for v in e:
                self.idx[v].add(e)
        self.Hc = Counter(len(e) for e in G_frozen)
        self.Sc = Counter()
        # home index: every live object is filed under its rarest node (by
        # degree in G). Objects inside a node set u are found by scanning
        # the home lists of u's nodes, which skips hub incidences.
        self.gdeg = {v: len(es) for v, es in self.idx.items()}
        self.home = defaultdict(set)
        for e in G_frozen:
            self.home[self._rare(e)].add(e)
        # static strict-descendant edges, via the nestedness DAG (rarest-node scan)
        eidx = defaultdict(list)
        for e in G_frozen:
            for v in e:
                eidx[v].append(e)
        self.eidx = eidx
        self.nestc = {}                                 # node set -> sizes of G-edges inside
        self.sidx = defaultdict(set)                    # node -> live SIMPLICES (not H edges)
        self.desc = defaultdict(list)                   # edge -> edges strictly inside it
        for f in G_frozen:
            v0 = min(f, key=lambda v: len(eidx[v]))
            for e in eidx[v0]:
                if len(e) > len(f) and f < e:
                    self.desc[e].append(f)

    def _rare(self, t):
        g = self.gdeg
        return min(t, key=lambda v: (g[v], str(v)))

    def nest_of(self, x):
        h = self.nestc.get(x)
        if h is None:                     # only G edges reach here uncached
            h = Counter(len(f) for f in self.desc.get(x, ()))
            h[len(x)] += 1
            self.nestc[x] = h
        return h

    def nest_union(self, a, b, u):
        """Sizes of all G-edges inside u = a|b: those inside the larger
        operand (cached) plus edges through the nodes u adds to it."""
        big = a if len(a) >= len(b) else b
        h = Counter(self.nest_of(big))
        if len(u) == len(big):
            return h
        seen = set()
        eidx = self.eidx
        for v in u:
            if v in big:
                continue
            for f in eidx[v]:
                if f in seen:
                    continue
                seen.add(f)
                if f <= u:
                    h[len(f)] += 1
        return h

    def nest_for(self, a, b, u, T, hist):
        """Sizes of all G-edges inside u. Edges inside u that the merge does
        not absorb are owned by live simplices t outside T with |t & u| >= 2,
        because H edges inside u are absorbed. If there are none,
        nest = hist exactly. Otherwise count those simplices' faces inside u,
        or fall back to the incidence scan when that is cheaper."""
        ov = Counter()
        sidx = self.sidx
        for v in u:
            for t in sidx.get(v, ()):
                if t not in T:
                    ov[t] += 1
        cands = [t for t, c in ov.items() if c >= 2]
        if not cands:
            return Counter(hist)
        work = sum(len(self.faces[t]) for t in cands)
        scan = sum(self.gdeg.get(v, 0) for v in u if v not in (a if len(a) >= len(b) else b))
        if work > 2 * scan:
            return self.nest_union(a, b, u)
        h = Counter(hist)
        for t in cands:
            for f in self.faces[t]:
                if f <= u:
                    h[len(f)] += 1
        return h

    def is_simplex(self, t):
        return t in self.simp

    def sub_objects(self, x):
        """Live objects strictly inside x. They are empty for a simplex (invariant)."""
        if x in self.simp:
            return ()
        out = set()
        for f in self.desc.get(x, ()):
            t = self.obox[f][0]
            if t is not x and t <= x:
                out.add(t)
        return out

    def cross_objects(self, a, b):
        """Live objects inside a|b that are inside neither a nor b; each must
        meet both a-b and b-a. Work is kept proportional to the SMALLER
        operand (plus incidence scans), never to a giant operand's size:
        small unions enumerate candidate node subsets directly, otherwise
        the incidence lists of the smaller difference side (or the home
        lists of a small union) are scanned."""
        if len(a) > len(b):
            a, b = b, a
        d1 = [v for v in a if v not in b]
        if not d1:
            return ()
        n_sh = len(a) - len(d1)
        if len(b) == n_sh:
            return ()
        nu = len(b) + len(d1)
        idx = self.idx
        c1 = sum(map(len, map(idx.__getitem__, d1)))
        side, est = d1, c1
        if len(b) <= 2000:
            d2 = [v for v in b if v not in a]
            c2 = sum(map(len, map(idx.__getitem__, d2)))
            if c2 < c1:
                side, est = d2, c2
            if nu <= 20:
                n_sub = (1 << nu) - (1 << len(a)) - (1 << len(b)) + (1 << n_sh)
                if 4 * n_sub < est:
                    faces = self.faces
                    shared = [v for v in a if v in b]

                    def subsets(lst):
                        r = [()]
                        for v in lst:
                            r = r + [x + (v,) for x in r]
                        return r
                    S1, S2, S0 = subsets(d1)[1:], subsets(d2)[1:], subsets(shared)
                    out = set()
                    for x1 in S1:
                        for x2 in S2:
                            for x0 in S0:
                                t = frozenset(x1 + x2 + x0)
                                if t in faces:
                                    out.add(t)
                    return out
        u = a | b
        if nu <= 256:
            home = self.home
            ch = sum(map(len, map(home.__getitem__, u)))
            if ch < est:
                cand = set().union(*(home[v] for v in u))
                return {t for t in cand if t <= u and not t <= a and not t <= b}
        cand = set().union(*(idx[v] for v in side))
        return {t for t in cand if t <= u}

    def plan(self, a, b):
        """Exact delta-L for merging a and b; returns (delta, T, hist)."""
        u = a | b
        T = {a, b}
        T.update(self.sub_objects(a))
        T.update(self.sub_objects(b))
        T.update(self.cross_objects(a, b))
        if u not in T and u in self.faces:   # u already exists as a live H edge
            T.add(u)
            T.update(self.sub_objects(u))
        N = self.N
        hist = Counter()
        dH = Counter()
        dS = Counter()
        d = 0.0
        for t in T:
            hist.update(self.hist[t])
            if t in self.simp:
                dS[len(t)] -= 1
                d -= self.simp[t]
            else:
                dH[len(t)] -= 1
        nu = len(u)
        dS[nu] += 1
        nest = self.nestc.get(u)
        if nest is None:
            nest = self.nest_for(a, b, u, T, hist)
        d += set_local_cost(nu, hist, nest)
        for j, c in dH.items():
            if c:
                h0 = self.Hc[j]
                d += _lcc(N, j, h0 + c) - _lcc(N, j, h0)
        for j, c in dS.items():
            if c:
                s0 = self.Sc[j]
                d += _lcc(N, j, s0 + c) - _lcc(N, j, s0)
        return d, T, (hist, nest)

    def apply(self, u, T, hn):
        """Execute a planned merge; returns an undo record. The face set and
        owner box of the operand with the most faces are reused (stolen),
        so the cost is proportional to the smaller operands."""
        hist, nest = hn
        self.nestc[u] = nest
        faces, idx, home = self.faces, self.idx, self.home
        big = max(T, key=lambda t: (len(faces[t]), _enclosing_key(t)))
        rec = []
        for t in T:
            rec.append((t, None if t is big else faces[t], self.hist[t],
                        self.simp.get(t), self.box[t]))
            if t in self.simp:
                self.Sc[len(t)] -= 1
                del self.simp[t]
                for v in t:
                    self.sidx[v].discard(t)
            else:
                self.Hc[len(t)] -= 1
            del self.hist[t]
            for v in t:
                idx[v].discard(t)
            home[self._rare(t)].discard(t)
        bigfaces = faces.pop(big)
        bigbox = self.box.pop(big)
        for t in T:
            if t is big:
                continue
            f = faces.pop(t)
            del self.box[t]
            bigfaces |= f
            obox = self.obox
            for e in f:
                obox[e] = bigbox
        bigbox[0] = u
        faces[u] = bigfaces
        self.box[u] = bigbox
        home[self._rare(u)].add(u)
        self.hist[u] = hist
        self.simp[u] = set_local_cost(len(u), hist, self.nestc[u])
        self.Sc[len(u)] += 1
        for v in u:
            idx[v].add(u)
            self.sidx[v].add(u)
        return (u, big, rec)

    def undo(self, record):
        u, big, rec = record
        faces, idx, home = self.faces, self.idx, self.home
        self.Sc[len(u)] -= 1
        del self.simp[u]
        bigfaces = faces.pop(u)
        bigbox = self.box.pop(u)
        del self.hist[u]
        for v in u:
            idx[v].discard(u)
            self.sidx[v].discard(u)
        home[self._rare(u)].discard(u)
        for t, f, hist, cost, tbox in rec:
            if t is not big:
                bigfaces -= f
                faces[t] = f
                self.box[t] = tbox
                for e in f:
                    self.obox[e] = tbox
        bigbox[0] = big
        faces[big] = bigfaces
        self.box[big] = bigbox
        for t, f, hist, cost, tbox in rec:
            home[self._rare(t)].add(t)
            self.hist[t] = hist
            if cost is None:
                self.Hc[len(t)] += 1
            else:
                self.simp[t] = cost
                self.Sc[len(t)] += 1
                for v in t:
                    self.sidx[v].add(t)
            for v in t:
                idx[v].add(t)

    def total(self):
        return sum(self.simp.values()) + set_global_cost(self.N, self.Hc, self.Sc)


def set_greedy(G_frozen, N, node_pair_cap=None, hub_cap=None, budget_s=None,
               logfile=None, seed=0, nucleation_cap=2000, nucleation_bits=64.0,
               max_rounds=4):
    """Exact greedy for the set-code objective. Returns (S, L_state).

    Phase 1 absorbs nested hyperedges collectively into their largest
    enclosing hyperedge. Phase 2 runs lazy best-first pair merges. Each
    phase ends with a NUCLEATION pass. Because S and H are coded as sets,
    the cost of a new simplex falls as more simplices of its size exist,
    so several moves can be jointly favourable while each is individually
    unfavourable (e.g. two disjoint triangles in a tiny graph). The pass
    applies the best near-miss candidates (delta < nucleation_bits) in
    ascending order of delta, re-scoring each one exactly. It keeps the
    prefix with the lowest cumulative L and undoes the rest, so it can
    only lower L. Rounds repeat until nothing improves (max_rounds)."""
    if node_pair_cap is None or node_pair_cap == 'auto':
        node_pair_cap = NODE_PAIR_CAP
    if hub_cap is None:
        hub_cap = HUB_CAP
    t0 = time.time()
    deadline = t0 + budget_s if budget_s else None
    st = _SetState(G_frozen, N)
    L = st.total()
    rng = random.Random(seed)
    _kc = {}

    def _key(t):
        k = _kc.get(t)
        if k is None:
            k = _kc[t] = (len(t), tuple(sorted(map(str, t))))
        return k
    TOL = 1e-9
    # A popped candidate whose re-scored delta is within STALE_SLACK bits of
    # the next heap entry is executed rather than re-queued. Every move
    # is still scored exactly; this only relaxes the best-first ORDER, so
    # global-count drift cannot set off a re-queue cascade.
    STALE_SLACK = 1.0

    def log(msg):
        if logfile:
            logfile.write(msg + f"  [{time.time() - t0:.1f}s]\n")
            logfile.flush()

    def nucleate(cands, label):
        """cands: iterable of (delta, a, b). Returns the L change (<= 0)."""
        cands = sorted((c for c in cands if c[0] < nucleation_bits),
                       key=lambda c: (c[0], _key(c[1]), _key(c[2])))[:nucleation_cap]
        undo_log, cum, best, best_len = [], 0.0, 0.0, 0
        for _, a, b in cands:
            if a not in st.faces or b not in st.faces:
                continue
            d, T, hist = st.plan(a, b)
            if not math.isfinite(d) or d >= nucleation_bits:
                continue
            undo_log.append(st.apply(a | b, T, hist))
            cum += d
            if cum < best - TOL:
                best, best_len = cum, len(undo_log)
        for rec in reversed(undo_log[best_len:]):
            st.undo(rec)
        if best_len:
            log(f"  nucleation ({label}): {best_len} joint moves, dL={best:.3f}")
        return best

    # ── phase 1: collective absorption into the largest enclosing edges ────
    def phase1():
        dL, near, n = 0.0, [], 0
        for p in sorted(G_frozen, key=lambda t: (-len(t), _key(t)[1])):
            if p not in st.faces or p in st.simp or not st.desc.get(p):
                continue
            if not st.sub_objects(p):
                continue
            d, T, hist = st.plan(p, p)
            if d < -TOL:
                st.apply(p, T, hist)
                dL += d
                n += 1
            elif d < nucleation_bits:
                near.append((d, p, p))
        log(f"set-code phase 1: {n} absorptions, |S|={len(st.simp)}")
        return dL, near

    # ── phase 2: lazy best-first pair merges ────────────────────────────────
    def phase2():
        heap, near = [], []
        cnt = 0
        partners = defaultdict(set)

        def push(a, b):
            nonlocal cnt
            d, _, _ = st.plan(a, b)
            if d < -TOL:
                cnt += 1
                heapq.heappush(heap, (d, cnt, a, b))
                partners[a].add(b)
                partners[b].add(a)
            elif d < nucleation_bits:
                near.append((d, a, b))

        seen = set()
        for v in sorted(st.idx, key=str):
            objs = sorted(st.idx[v], key=_key)
            d_ = len(objs)
            if d_ < 2:
                continue
            if d_ * (d_ - 1) // 2 > node_pair_cap:
                r = random.Random(zlib.crc32(repr(v).encode('utf8', 'replace')))
                pairs = set()
                while len(pairs) < node_pair_cap:
                    i, k = r.randrange(d_), r.randrange(d_)
                    if i != k:
                        pairs.add((min(i, k), max(i, k)))
                pairs = sorted(pairs)
            else:
                pairs = [(i, k) for i in range(d_) for k in range(i + 1, d_)]
            n_eval = len(seen)
            if n_eval // 200000 != (n_eval + len(pairs)) // 200000:
                log(f"  seeding: {n_eval} pairs evaluated, heap={len(heap)}")
            for i, k in pairs:
                a, b = objs[i], objs[k]
                key = (a, b) if _key(a) <= _key(b) else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                push(a, b)
            if deadline and time.time() > deadline:
                log("BUDGET: seeding truncated")
                break
        seen = None

        dL, n2 = 0.0, 0
        while heap:
            if deadline and time.time() > deadline:
                log("BUDGET: phase 2 stopped early")
                break
            _, _, a, b = heapq.heappop(heap)
            if a not in st.faces or b not in st.faces:
                continue
            d, T, hist = st.plan(a, b)
            if d >= -TOL:
                if d < nucleation_bits:
                    near.append((d, a, b))
                continue
            if heap and d > heap[0][0] + STALE_SLACK:   # stale: re-queue at its true value
                cnt += 1
                heapq.heappush(heap, (d, cnt, a, b))
                continue
            u = a | b
            old_partners = set()
            for t in T:
                old_partners |= partners.pop(t, set())
            st.apply(u, T, hist)
            dL += d
            n2 += 1
            if n2 % 2000 == 0:
                log(f"  phase 2: {n2} merges, |S|={len(st.simp)}, heap={len(heap)}, "
                    f"max|s|={len(u)}")
            # re-pair: earlier partners of the absorbed objects, plus the
            # neighbours of u with the largest overlap (capped at hub_cap)
            cand = {t for t in old_partners if t in st.faces and t != u}
            big_op = max(T, key=len)
            new_nodes = set()
            for t in T:
                if t is not big_op:
                    new_nodes.update(v for v in t if v not in big_op)
            ov = Counter()
            scanned = 0
            scan = sorted(new_nodes, key=str)
            if len(big_op) <= 64:
                scan += sorted(big_op, key=str)
            for v in scan:
                for t in st.idx[v]:
                    if t is not u:
                        ov[t] += 1
                scanned += len(st.idx[v])
                if scanned > 50 * hub_cap:
                    break
            cand.update(sorted(ov, key=lambda t: (-ov[t], _key(t)))[:hub_cap])
            if len(cand) > hub_cap:     # per-union re-pairing budget
                cand = set(sorted(cand, key=lambda t: (-ov.get(t, 0), _key(t)))[:hub_cap])
            for t in sorted(cand, key=_key):
                push(u, t)
        log(f"set-code phase 2: {n2} merges, |S|={len(st.simp)}")
        return dL, near

    for rnd in range(max_rounds):
        L_start = L
        d1, near1 = phase1()
        L += d1
        L += nucleate(near1, 'phase 1')
        if deadline and time.time() > deadline:
            break
        d2, near2 = phase2()
        L += d2
        L += nucleate(near2, 'phase 2')
        log(f"round {rnd + 1}: L={L:.3f}")
        if L > L_start - 1e-6 or (deadline and time.time() > deadline):
            break

    chk = st.total()
    if abs(chk - L) > 1e-6 * max(1.0, abs(L)):
        raise RuntimeError(f"set-code bookkeeping drift: {L} vs {chk}")
    return list(st.simp), chk


def _set_greedy_optimizer(G_list, logpath=os.devnull, N=None, budget_s=None,
                          node_pair_cap='auto', hub_cap=None, **_ignored):
    G_frozen = frozenset(frozenset(e) for e in G_list)
    if N is None:
        N = len(set(v for e in G_frozen for v in e))
    t0 = time.time()
    with open(logpath, 'w') as lf:
        L0 = set_baseline_DL(G_frozen, N)
        lf.write(f"[set code] N={N}  E={len(G_frozen)}  L0={L0:.3f}\n")
        S, Lstate = set_greedy(G_frozen, N, node_pair_cap=node_pair_cap,
                               hub_cap=hub_cap, budget_s=budget_s, logfile=lf)
        DL = set_description_length(G_frozen, S, N)
        lf.write(f"state L={Lstate:.3f}  assignment-rule L={DL:.3f}\n")
        if DL >= L0:
            S, DL = [], L0
        lf.write(f"Final: |S|={len(S)}  DL={DL:.3f}  eta={DL / L0:.4f}\n")
    last_run_info.clear()
    last_run_info.update(mode='set-exact', objective='set', requested_mode='exact',
                         E=len(G_frozen), S0=len(S), trigger=None,
                         node_pair_cap=node_pair_cap, hub_cap=hub_cap,
                         runtime_s=time.time() - t0)
    return _finalise(G_frozen, S, DL, L0)




greedy_optimizer = _set_greedy_optimizer
description_length = set_description_length
baseline_DL = set_baseline_DL


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
