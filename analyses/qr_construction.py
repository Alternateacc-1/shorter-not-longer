"""Binary quadratic-residue codes from first principles, no tables.

For a prime p = 8m +- 1 the binary QR code of length p is the cyclic code with idempotent
e_Q(x) = sum_{r in Q} x^r (p = 8m-1) or 1 + e_Q(x) (p = 8m+1), Q the nonzero squares mod p
(MacWilliams & Sloane, ch. 16 sec. 3). Here the generator is computed as g = gcd(e(x), x^p - 1)
over GF(2) for every candidate idempotent, and a candidate is accepted as THE QR code only if
  * e(x)^2 = e(x) mod x^p - 1 (an idempotent), g divides x^p - 1, deg g = (p-1)/2, k = (p+1)/2;
  * the code is invariant under every multiplier x -> x^a, a in Q, and under none with a a
    non-residue (the defining symmetry of a QR code: mu_a, a in N, maps it to the other QR code).
The minimum distance and the whole weight distribution come from enumerating all 2^k codewords
(meet in the middle: two half-dimension tables XORed), and independently, for k <= 22, from
pipeline.bch_codes.min_distance, and for every k from min_weight() below, a second
enumeration with a different split.
Integers encode polynomials and words: bit j = coefficient of x^j = coordinate j.
"""
import numpy as np


def min_weight(rows):
    """Minimum weight over all 2^k - 1 nonzero codewords, by a second enumeration
    (split table: 2^min(k,20) x 2^(k-20)). Returns (minimum weight, 2^k)."""
    lo, hi = rows[:20], rows[20:]
    table = np.zeros(1, dtype=np.uint64)
    for r in lo:
        table = np.concatenate([table, table ^ np.uint64(r)])
    best, count = 10 ** 9, 0
    for j in range(1 << len(hi)):
        h = np.uint64(0)
        for b, r in enumerate(hi):
            if j >> b & 1:
                h ^= np.uint64(r)
        w = np.bitwise_count(table ^ h)
        if j == 0:
            w = w[1:]
        best = min(best, int(w.min())); count += len(w)
    assert count == 2 ** len(rows) - 1
    return best, count + 1


def residues(p):
    return sorted({(x * x) % p for x in range(1, p)})


def pdeg(a):
    return a.bit_length() - 1


def pmod(a, b):
    db = pdeg(b)
    while a and pdeg(a) >= db:
        a ^= b << (pdeg(a) - db)
    return a


def pmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1; b >>= 1
    return r


def pgcd(a, b):
    while b:
        a, b = b, pmod(a, b)
    return a


def mult(c, a, p):
    """c(x) -> c(x^a) mod x^p - 1 (a coordinate permutation i -> a*i mod p)."""
    out = 0
    for i in range(p):
        if (c >> i) & 1:
            out |= 1 << ((a * i) % p)
    return out


def cyclic_rows(g, p):
    return [g << i for i in range(p - pdeg(g))]


def weight_distribution(rows, n, split=None):
    """A_w for w = 0..n over all 2^k codewords, by two half tables (exhaustive)."""
    k = len(rows); k1 = k // 2 if split is None else split

    def table(rs):
        t = np.zeros(1, dtype=np.uint64)
        for r in rs:
            t = np.concatenate([t, t ^ np.uint64(r)])
        return t
    T1, T2 = table(rows[:k1]), table(rows[k1:])
    A = np.zeros(n + 1, dtype=np.int64)
    for x in T2:
        A += np.bincount(np.bitwise_count(T1 ^ x), minlength=n + 1)
    assert A.sum() == 2 ** k and A[0] == 1
    return A


def qr_code(p):
    assert p % 8 in (1, 7), "binary QR codes need p = 8m +- 1"
    Q = residues(p); Nn = sorted(set(range(1, p)) - set(Q))
    xp1 = (1 << p) | 1                                     # x^p - 1 = x^p + 1 over GF(2)
    eQ = sum(1 << r for r in Q)
    cands = {"e_Q": eQ, "1+e_Q": eQ ^ 1, "e_N": sum(1 << r for r in Nn), "1+e_N": sum(1 << r for r in Nn) ^ 1}
    found = []
    for name, e in cands.items():
        if pmod(pmul(e, e), xp1) != e:
            continue
        g = pgcd(xp1, e)
        if pdeg(g) != (p - 1) // 2:
            continue
        found.append((name, e, g))
    name, e, g = next(f for f in found if f[0].endswith("e_Q"))   # the Q-side code (the N-side one is equivalent)
    assert pmod(xp1, g) == 0
    rows = cyclic_rows(g, p); k = len(rows)
    inv_Q = all(pmod(mult(g, a, p), g) == 0 for a in Q)
    inv_N = any(pmod(mult(g, a, p), g) == 0 for a in Nn)
    A = weight_distribution(rows, p)
    d = int(next(w for w in range(1, p + 1) if A[w]))
    return {"p": p, "n": p, "k": k, "parity": p - k, "d": d, "t": (d - 1) // 2, "idempotent": name,
            "generator_poly": g, "generator_hex": hex(g), "generator_exponents": [i for i in range(p) if (g >> i) & 1],
            "residues": Q, "candidates_found": [f[0] for f in found], "invariant_under_all_Q_multipliers": inv_Q,
            "invariant_under_some_N_multiplier": inv_N, "weight_distribution": {int(w): int(a) for w, a in enumerate(A) if a},
            "min_weight_codewords": int(A[d]), "rows": rows}
